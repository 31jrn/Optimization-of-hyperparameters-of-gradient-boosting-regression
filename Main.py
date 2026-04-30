"""
Программа: Оптимизация гиперпараметров градиентного бустинга (регрессия)
Требования:
- Загрузка данных из Excel (любое количество строк)
- Разделение на train/test 80/20
- Режимы: без кросс-валидации и с 5-fold CV
- Методы оптимизации: сеточный (Grid), случайный (Random), байесовский (Bayes)
- Оценка: RMSE, MAPE, время
- Вывод итоговых таблиц
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import optuna
import logging


def load_data(file_path, x_cols=None, y_col=None):
    """
    Загружает данные из Excel.
    По умолчанию:
        - целевая переменная: первый столбец (индекс 0)
        - признаки: столбцы с индексами 1..10 (всего 10)
    Предполагается, что первая строка - заголовки.
    """
    df = pd.read_excel(file_path, header=0)
    df = df.iloc[:, :11]
    print("Первые 5 строк загруженных данных:")
    print(df.head())
    print(f"Все столбцы: {df.columns.tolist()}")

    if x_cols is None:
        # Берём столбцы с 1 по 10 (индексы 1..10) - всего 10 признаков
        x_cols = df.columns[1:11].tolist()
    if y_col is None:
        # Целевая - первый столбец (индекс 0)
        y_col = df.columns[0]

    X = df[x_cols].values
    y = df[y_col].values

    # Принудительно преобразуем y в float, если он ещё не число
    if not np.issubdtype(y.dtype, np.number):
        try:
            # Заменяем запятые на точки, если есть
            y = np.array([str(val).replace(",", ".") for val in y])
            y = y.astype(float)
        except ValueError as e:
            raise ValueError(
                f"Целевая переменная '{y_col}' содержит нечисловые значения, которые не удалось преобразовать: {e}"
            )

    print(f"Загружено: X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"Имена признаков: {x_cols}")
    print(f"Целевая переменная: {y_col}")
    return X, y


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Разделяет данные на train и test.
    Для режима без CV дополнительно выделяет валидационную выборку из train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, shuffle=True
    )
    return (X_train, X_test, y_train, y_test), (
        X_train_train,
        X_train_val,
        y_train_train,
        y_train_val,
    )


def _objective(trial, X_train, y_train, X_val, y_val, metric_func):
    # 1️⃣ РЕГУЛЯРИЗАЦИЯ ПОИСКОВОГО ПРОСТРАНСТВА
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "gamma": trial.suggest_float("gamma", 0.1, 2.0),
        "lambda": trial.suggest_float("lambda", 0.5, 3.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": 1000,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "early_stopping_rounds": 50,  # 🟢 ПЕРЕНЕСЕНО В КОНСТРУКТОР
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],  # eval_set остаётся в fit()
        verbose=False,  # early_stopping_rounds убран отсюда
    )

    # XGBoost 2.x автоматически подставляет best_iteration, если ранняя остановка сработала
    best_n = (
        model.best_iteration if model.best_iteration is not None else model.n_estimators
    )
    trial.set_user_attr("best_n_estimators", int(best_n))

    val_preds = model.predict(X_val)
    return metric_func(y_val, val_preds)


def optimize_xgb_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=100,
    metric_func=mean_squared_error,
    direction="minimize",
):
    """
    Запуск Optuna оптимизации для XGBoost.
    Возвращает: best_params, best_value, best_n_estimators
    """
    study = optuna.create_study(
        direction=direction,
        study_name="xgb_hyperopt",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val, metric_func),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value
    best_n_estimators = study.best_trial.user_attrs["best_n_estimators"]

    print(f"\n✅ Лучшие гиперпараметры: {best_params}")
    print(f"🌳 Оптимальное число деревьев (early stopping): {best_n_estimators}")
    print(f"📉 Лучшая метрика на валидации: {best_value:.4f}")

    return best_params, best_value, best_n_estimators


def get_model():
    """Возвращает базовую модель XGBoost Regressor."""
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)


def get_param_grid():

    #  Определяет сетку для GridSearch (дискретные значения)
    #  и пространство для Random / Bayes (распределения).

    # Для GridSearch используем небольшой набор дискретных значений
    grid_params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Для RandomizedSearchCV используем более широкий дискретный набор
    random_params = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # Для BayesSearchCV задаём непрерывные диапазоны
    bayes_params = {
        "n_estimators": Integer(50, 200),
        "max_depth": Integer(3, 8),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0),
    }

    return grid_params, random_params, bayes_params


def evaluate_model(model, X_test, y_test):
    """Возвращает RMSE и MAPE на тестовых данных."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # в процентах
    return rmse, mape


def main():
    # Параметры (можно менять)
    DATA_PATH = "Мазур-данные для регрессии.xlsx"  # путь к файлу Excel
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2  # доля валидации от train (для режима без CV)
    RANDOM_STATE = 42
    N_ITER_SEARCH = 20  # количество итераций для random и bayes

    # 1. Загрузка
    print("Загрузка данных...")
    X, y = load_data(DATA_PATH)

    # 2. Разделение
    (
        (X_train, X_test, y_train, y_test),
        (X_train_train, X_train_val, y_train_train, y_train_val),
    ) = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train_train size: {len(X_train_train)}, Train_val size: {len(X_train_val)}")

    # 3. Оптимизация через Optuna (с ранней остановкой и регуляризацией пространства)
    # Для валидации используем X_train_val, y_train_val (они возвращаются из split_data)
    best_params, _, best_n_estimators = optimize_xgb_optuna(
        X_train_train,
        y_train_train,
        X_train_val,
        y_train_val,
        n_trials=50,  # Для теста хватит 50, для продакшена ставьте 100-200
    )

    final_model = xgb.XGBRegressor(
        **best_params, n_estimators=best_n_estimators, random_state=RANDOM_STATE
    )
    final_model.fit(X_train, y_train)  # Финальное обучение на полном трейне

    # Возвращаем параметры для Grid/Random/Bayes (раскомментируем вызов!)
    grid_params, random_params, bayes_params = get_param_grid()

    # 4. Результаты будем хранить в словаре
    results = {
        "no_cv": {"grid": {}, "random": {}, "bayes": {}},
        "with_cv": {"grid": {}, "random": {}, "bayes": {}},
    }

    print("\n=== Режим без кросс-валидации ===")
    # Формируем единый массив из train_train и train_val для подачи в *SearchCV,
    # и задаём пользовательский cv, который представляет собой один фиксированный сплит.
    train_indices = np.arange(len(X_train_train))
    val_indices = np.arange(len(X_train_train), len(X_train_train) + len(X_train_val))
    custom_cv = [
        (train_indices, val_indices)
    ]  # список из одного кортежа (train_idx, val_idx)

    X_train_full = np.vstack([X_train_train, X_train_val])
    y_train_full = np.concatenate([y_train_train, y_train_val])

    # Сеточный поиск (без CV)
    print("Сеточный поиск (без CV)...")
    start = time.time()
    model = get_model()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_params,
        scoring="neg_root_mean_squared_error",
        cv=custom_cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_full, y_train_full)
    time_grid = time.time() - start
    best_params_grid = grid_search.best_params_
    print(f"Лучшие параметры (grid): {best_params_grid}")

    # Обучаем финальную модель на всём train с лучшими параметрами
    final_model_grid = xgb.XGBRegressor(
        **best_params_grid, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_grid.fit(X_train, y_train)
    rmse_grid, mape_grid = evaluate_model(final_model_grid, X_test, y_test)
    results["no_cv"]["grid"] = {"rmse": rmse_grid, "mape": mape_grid, "time": time_grid}

    # Случайный поиск (без CV)
    print("Случайный поиск (без CV)...")
    start = time.time()
    model = get_model()
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_params,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=custom_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    random_search.fit(X_train_full, y_train_full)
    time_random = time.time() - start
    best_params_random = random_search.best_params_
    print(f"Лучшие параметры (random): {best_params_random}")

    final_model_random = xgb.XGBRegressor(
        **best_params_random, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_random.fit(X_train, y_train)
    rmse_random, mape_random = evaluate_model(final_model_random, X_test, y_test)
    results["no_cv"]["random"] = {
        "rmse": rmse_random,
        "mape": mape_random,
        "time": time_random,
    }

    # Байесовский поиск (без CV)
    print("Байесовский поиск (без CV)...")
    start = time.time()
    model = get_model()
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=bayes_params,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=custom_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    bayes_search.fit(X_train_full, y_train_full)
    time_bayes = time.time() - start
    best_params_bayes = bayes_search.best_params_
    print(f"Лучшие параметры (bayes): {best_params_bayes}")

    final_model_bayes = xgb.XGBRegressor(
        **best_params_bayes, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_bayes.fit(X_train, y_train)
    rmse_bayes, mape_bayes = evaluate_model(final_model_bayes, X_test, y_test)
    results["no_cv"]["bayes"] = {
        "rmse": rmse_bayes,
        "mape": mape_bayes,
        "time": time_bayes,
    }

    print("\n=== Режим с кросс-валидацией (5-fold) ===")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Сеточный поиск с CV
    print("Сеточный поиск (с CV)...")
    start = time.time()
    model = get_model()
    grid_search_cv = GridSearchCV(
        estimator=model,
        param_grid=grid_params,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search_cv.fit(X_train, y_train)
    time_grid_cv = time.time() - start
    best_params_grid_cv = grid_search_cv.best_params_
    print(f"Лучшие параметры (grid CV): {best_params_grid_cv}")

    final_model_grid_cv = xgb.XGBRegressor(
        **best_params_grid_cv, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_grid_cv.fit(X_train, y_train)
    rmse_grid_cv, mape_grid_cv = evaluate_model(final_model_grid_cv, X_test, y_test)
    results["with_cv"]["grid"] = {
        "rmse": rmse_grid_cv,
        "mape": mape_grid_cv,
        "time": time_grid_cv,
    }

    # Случайный поиск с CV
    print("Случайный поиск (с CV)...")
    start = time.time()
    model = get_model()
    random_search_cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_params,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    random_search_cv.fit(X_train, y_train)
    time_random_cv = time.time() - start
    best_params_random_cv = random_search_cv.best_params_
    print(f"Лучшие параметры (random CV): {best_params_random_cv}")

    final_model_random_cv = xgb.XGBRegressor(
        **best_params_random_cv, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_random_cv.fit(X_train, y_train)
    rmse_random_cv, mape_random_cv = evaluate_model(
        final_model_random_cv, X_test, y_test
    )
    results["with_cv"]["random"] = {
        "rmse": rmse_random_cv,
        "mape": mape_random_cv,
        "time": time_random_cv,
    }

    # Байесовский поиск с CV
    print("Байесовский поиск (с CV)...")
    start = time.time()
    model = get_model()
    bayes_search_cv = BayesSearchCV(
        estimator=model,
        search_spaces=bayes_params,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    bayes_search_cv.fit(X_train, y_train)
    time_bayes_cv = time.time() - start
    best_params_bayes_cv = bayes_search_cv.best_params_
    print(f"Лучшие параметры (bayes CV): {best_params_bayes_cv}")

    final_model_bayes_cv = xgb.XGBRegressor(
        **best_params_bayes_cv, objective="reg:squarederror", random_state=RANDOM_STATE
    )
    final_model_bayes_cv.fit(X_train, y_train)
    rmse_bayes_cv, mape_bayes_cv = evaluate_model(final_model_bayes_cv, X_test, y_test)
    results["with_cv"]["bayes"] = {
        "rmse": rmse_bayes_cv,
        "mape": mape_bayes_cv,
        "time": time_bayes_cv,
    }

    columns = [
        "Характеристика",
        "Сеточный поиск",
        "Случайный поиск",
        "Байесовский поиск",
    ]

    # Таблица для режима без CV
    data_no_cv = [
        [
            "RMSE",
            f"{results['no_cv']['grid']['rmse']:.4f}",
            f"{results['no_cv']['random']['rmse']:.4f}",
            f"{results['no_cv']['bayes']['rmse']:.4f}",
        ],
        [
            "MAPE (%)",
            f"{results['no_cv']['grid']['mape']:.2f}",
            f"{results['no_cv']['random']['mape']:.2f}",
            f"{results['no_cv']['bayes']['mape']:.2f}",
        ],
        [
            "Время (с)",
            f"{results['no_cv']['grid']['time']:.2f}",
            f"{results['no_cv']['random']['time']:.2f}",
            f"{results['no_cv']['bayes']['time']:.2f}",
        ],
    ]
    df_no_cv = pd.DataFrame(data_no_cv, columns=columns)

    # Таблица для режима с CV
    data_with_cv = [
        [
            "RMSE",
            f"{results['with_cv']['grid']['rmse']:.4f}",
            f"{results['with_cv']['random']['rmse']:.4f}",
            f"{results['with_cv']['bayes']['rmse']:.4f}",
        ],
        [
            "MAPE (%)",
            f"{results['with_cv']['grid']['mape']:.2f}",
            f"{results['with_cv']['random']['mape']:.2f}",
            f"{results['with_cv']['bayes']['mape']:.2f}",
        ],
        [
            "Время (с)",
            f"{results['with_cv']['grid']['time']:.2f}",
            f"{results['with_cv']['random']['time']:.2f}",
            f"{results['with_cv']['bayes']['time']:.2f}",
        ],
    ]
    df_with_cv = pd.DataFrame(data_with_cv, columns=columns)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ БЕЗ КРОСС-ВАЛИДАЦИИ")
    print("=" * 60)
    print(df_no_cv.to_string(index=False))

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ С КРОСС-ВАЛИДАЦИЕЙ (5-FOLD)")
    print("=" * 60)
    print(df_with_cv.to_string(index=False))


if __name__ == "__main__":
    main()
