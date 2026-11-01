from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from xgboost import XGBRegressor


FUTURE_DERIVED_MARKERS = ("_target_", "_pred", "_savgol")
BASELINE_MODELS = {"persistence_state", "persistence_horizon_lag", "seasonal_24h_lag"}
ML_MODELS = {"LinearRegression", "ElasticNet", "RandomForest", "XGBoost", "CatBoost"}
OPTUNA_MODELS = {"ElasticNet", "RandomForest", "XGBoost", "CatBoost"}


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска из командной строки."""
    parser = argparse.ArgumentParser(description="Run leak-aware ML baselines and Optuna models on feature CSVs.")
    parser.add_argument("--config", default="configs/experiments/baseline_v0.1.json")
    parser.add_argument("--station", default=None, help="Optional single station override.")
    parser.add_argument("--horizon", default=None, help="Optional single horizon override, for example 1h.")
    parser.add_argument("--no-automl", action="store_true", help="Disable Optuna even if enabled in the config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print planned run size without training.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Загружает конфигурацию эксперимента или сбора данных из файла."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_name(value: str) -> str:
    """Преобразует строку в безопасное имя для файлов и папок."""
    return value.replace(" ", "").replace("/", "_").replace("-", "_")


def utc_run_stamp() -> str:
    """Создает UTC-метку времени для имени запуска."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def output_prefix(config: dict[str, Any], station: str) -> str:
    """Формирует общий префикс выходных файлов эксперимента."""
    experiment_id = safe_name(str(config.get("experiment_id", "experiment")))
    run_started = safe_name(str(config.get("run_started_utc") or utc_run_stamp()))
    return f"{safe_name(station)}_{experiment_id}_{run_started}"


def horizon_to_hours(horizon: str) -> float:
    """Переводит горизонт прогноза в часы."""
    return pd.Timedelta(horizon) / pd.Timedelta(hours=1)


def latitude_zone(latitude: float | int | None) -> str | None:
    """Определяет широтную зону по широте станции."""
    if latitude is None or not np.isfinite(latitude):
        return None
    lat = float(latitude)
    if lat <= -60:
        return "S_high"
    if lat <= -30:
        return "S_mid"
    if lat < 30:
        return "Low"
    if lat < 60:
        return "N_mid"
    return "N_high"


def station_context(station: str, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Формирует контекст станции для метрик и отчетов."""
    context: dict[str, Any] = {"station": station}
    if frame is not None and not frame.empty:
        for column in ["latitude", "longitude", "geomagnetic_latitude", "geomagnetic_longitude"]:
            if column in frame.columns:
                value = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not value.empty:
                    context[column] = float(value.iloc[0])
    if "latitude" not in context:
        metadata_path = Path("configs/stations_metadata.csv")
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            row = metadata[metadata["station"] == station]
            if not row.empty:
                for column in ["country", "location", "latitude", "longitude"]:
                    if column in row.columns and pd.notna(row.iloc[0][column]):
                        context[column] = row.iloc[0][column].item() if hasattr(row.iloc[0][column], "item") else row.iloc[0][column]
    context["latitude_zone"] = latitude_zone(context.get("latitude"))
    return context


def enrich_rows(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Добавляет к строкам метрик контекст станции и запуска."""
    enriched = []
    for row in rows:
        enriched.append({**{k: v for k, v in context.items() if k != "station"}, **row})
    return enriched


def as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Преобразует значение времени в UTC timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_stations(config: dict[str, Any], station_override: str | None = None) -> list[str]:
    """Определяет список станций для запуска эксперимента."""
    if station_override:
        return [station_override]
    configured = config.get("stations", [])
    if configured == "all":
        station_dir = Path(config["feature_dir"]) / "stations"
        return sorted(path.name.removesuffix("_features.csv") for path in station_dir.glob("*_features.csv"))
    return list(configured)


def iter_test_days(config: dict[str, Any]) -> list[pd.Timestamp]:
    """Итерирует тестовые дни внутри заданного периода."""
    start = as_utc_timestamp(config.get("ml_date_start", config["test_start"]))
    end = as_utc_timestamp(config.get("ml_date_end", config["test_end"]))
    if end <= start:
        raise ValueError("test_end must be after test_start.")
    return list(pd.date_range(start, end - pd.Timedelta(days=1), freq="1D"))


def split_walk_forward_window(
    frame: pd.DataFrame,
    *,
    current_day: pd.Timestamp,
    train_days: int,
    forecast_horizon: str,
    dataset_step: str,
    time_column: str = "time_utc",
    validation_days: int | None = None,
    test_h: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Формирует train/test-разбиение для walk-forward проверки."""
    step = pd.Timedelta(dataset_step)
    forecast_delta = pd.Timedelta(forecast_horizon)
    if forecast_delta < step or forecast_delta % step != pd.Timedelta(0):
        raise ValueError(f"forecast_horizon {forecast_horizon!r} is not compatible with dataset_step {dataset_step!r}.")

    time = frame[time_column]
    test_start = as_utc_timestamp(current_day)
    if validation_days is None:
        test_end = test_start + pd.Timedelta(days=1) - step
        train_end = test_start - step
        train_start = train_end - pd.Timedelta(days=train_days) + step
        # Последние точки train отсекаются: их target на выбранном горизонте уже попал бы в тестовый день.
        safe_train_end = train_end - forecast_delta
        train = frame[(time >= train_start) & (time <= safe_train_end)].copy()
        validation = pd.DataFrame(columns=frame.columns)
        test = frame[(time >= test_start) & (time <= test_end)].copy()
        bounds = {
            "train_start": train_start,
            "train_end": train_end,
            "safe_train_end": safe_train_end,
            "test_start": test_start,
            "test_end": test_end,
        }
        return train, validation, test, bounds

    resolved_test_h = int(test_h or 24)
    test_delta = pd.Timedelta(hours=resolved_test_h)
    if test_delta < step:
        raise ValueError(f"test_h must be at least one dataset_step ({step}).")
    test_end = test_start + test_delta - step
    # Валидация ставится перед тестом с разрывом на горизонт прогноза, чтобы подбор не видел будущие ответы.
    val_end = test_start - forecast_delta - step
    val_start = val_end - pd.Timedelta(days=int(validation_days)) + step
    train_end = val_start - step
    train_start = train_end - pd.Timedelta(days=train_days) + step
    # safe_train_end защищает и validation: target последних train-строк не должен попадать в validation.
    safe_train_end = train_end - forecast_delta
    train = frame[(time >= train_start) & (time <= safe_train_end)].copy()
    validation = frame[(time >= val_start) & (time <= val_end)].copy()
    test = frame[(time >= test_start) & (time <= test_end)].copy()
    bounds = {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "validation_label_start": val_start + forecast_delta,
        "validation_label_end": val_end + forecast_delta,
        "test_start": test_start,
        "test_end": test_end,
    }
    return train, validation, test, bounds


def split_fixed_year(
    frame: pd.DataFrame,
    *,
    forecast_horizon: str,
    dataset_step: str,
    config: dict[str, Any],
    time_column: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Формирует train/test-разбиение по фиксированному годовому интервалу."""
    step = pd.Timedelta(dataset_step)
    forecast_delta = pd.Timedelta(forecast_horizon)
    if forecast_delta < step or forecast_delta % step != pd.Timedelta(0):
        raise ValueError(f"forecast_horizon {forecast_horizon!r} is not compatible with dataset_step {dataset_step!r}.")

    train_start = as_utc_timestamp(config["train_start"])
    train_end = as_utc_timestamp(config["train_end"]) - step
    val_start = as_utc_timestamp(config["validation_start"])
    requested_val_end = as_utc_timestamp(config["validation_end"]) - step
    test_start = as_utc_timestamp(config["test_start"])
    test_end = as_utc_timestamp(config["test_end"]) - step

    if not (train_start <= val_start <= requested_val_end <= train_end < test_start <= test_end):
        raise ValueError("Expected validation inside train period and test after train period for fixed_year_split.")

    safe_train_end = val_start - forecast_delta - step
    val_end = min(requested_val_end, train_end - forecast_delta)
    time = frame[time_column]
    train_core = frame[(time >= train_start) & (time <= safe_train_end)].copy()
    validation = frame[(time >= val_start) & (time <= val_end)].copy()
    test = frame[(time >= test_start) & (time <= test_end)].copy()
    bounds = {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "requested_validation_end": requested_val_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    return train_core, validation, test, bounds


def split_frames(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    horizon: str,
    current_day: pd.Timestamp | None = None,
    train_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Выбирает способ разбиения данных на обучение и тест."""
    mode = config.get("evaluation_mode", "fixed_year_split")
    if mode == "fixed_year_split":
        return split_fixed_year(
            frame,
            forecast_horizon=horizon,
            dataset_step=config["dataset_step"],
            config=config,
        )
    if mode == "walk_forward_daily":
        if current_day is None or train_days is None:
            raise ValueError("walk_forward_daily requires current_day and train_days.")
        automl = config.get("automl", {})
        use_validation = bool(automl.get("enabled", False))
        return split_walk_forward_window(
            frame,
            current_day=current_day,
            train_days=train_days,
            forecast_horizon=horizon,
            dataset_step=config["dataset_step"],
            validation_days=int(automl.get("val_days", 1)) if use_validation else None,
            test_h=int(automl.get("test_h", 24)) if use_validation else None,
        )
    raise ValueError(f"Unsupported evaluation_mode: {mode!r}")


def split_iterations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Создает последовательность итераций обучения и проверки."""
    if config.get("evaluation_mode", "fixed_year_split") == "fixed_year_split":
        return [{"split_id": "fixed_2024_train_2025_test", "train_days": None, "test_day": None}]
    return [
        {"split_id": f"{day.date().isoformat()}_{train_days}d", "train_days": int(train_days), "test_day": day}
        for train_days in config.get("window_list", config.get("walk_forward_train_days", []))
        for day in iter_test_days(config)
    ]


def numeric_feature_pool(frame: pd.DataFrame, target: str) -> list[str]:
    """Возвращает числовые признаки, доступные для обучения."""
    excluded = {"hour", "month", "doy"}
    target_prefix = f"{target}_target_"
    columns = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column in excluded:
            continue
        if column.startswith(target_prefix) or any(marker in column for marker in FUTURE_DERIVED_MARKERS):
            continue
        columns.append(column)
    return columns


def select_features(train: pd.DataFrame, feature_pool: list[str], min_coverage: float) -> list[str]:
    """Выбирает признаки с достаточным покрытием данных."""
    # Признак берется в модель только при достаточной заполненности в train-окне.
    return [feature for feature in feature_pool if feature in train.columns and train[feature].notna().mean() >= min_coverage]


def metric_record(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Рассчитывает метрики качества прогноза."""
    valid = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    if valid.empty:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan, "corr": math.nan, "bias": math.nan, "mape": math.nan}
    error = valid["predicted"] - valid["actual"]
    mae = float(mean_absolute_error(valid["actual"], valid["predicted"]))
    rmse = float(np.sqrt(mean_squared_error(valid["actual"], valid["predicted"])))
    r2 = float(r2_score(valid["actual"], valid["predicted"])) if len(valid) > 1 else math.nan
    corr = float(valid["actual"].corr(valid["predicted"])) if len(valid) > 1 else math.nan
    bias = float(error.mean())
    nonzero_actual = valid["actual"].abs() > 1e-12
    mape = float((error.loc[nonzero_actual].abs() / valid.loc[nonzero_actual, "actual"].abs()).mean() * 100) if nonzero_actual.any() else math.nan
    return {"n": int(len(valid)), "mae": mae, "rmse": rmse, "r2": r2, "corr": corr, "bias": bias, "mape": mape}


def make_baseline_predictions(test: pd.DataFrame, model_name: str, target: str, horizon: str) -> pd.Series:
    """Строит простые baseline-прогнозы для сравнения."""
    if model_name == "persistence_state":
        return pd.to_numeric(test.get(f"{target}_state"), errors="coerce")
    if model_name == "persistence_horizon_lag":
        return pd.to_numeric(test.get(f"{target}_lag_{safe_name(horizon)}"), errors="coerce")
    if model_name == "seasonal_24h_lag":
        return pd.to_numeric(test.get(f"{target}_lag_24h"), errors="coerce")
    raise ValueError(f"Unsupported baseline model: {model_name}")


def default_model_params(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Возвращает параметры моделей по умолчанию."""
    return dict(config.get("model_params", {}).get(model_name, {}))


def make_estimator(model_name: str, config: dict[str, Any], params: dict[str, Any] | None = None) -> object:
    """Создает объект модели машинного обучения по имени."""
    model_params = default_model_params(model_name, config)
    if params:
        model_params.update(params)
    if model_name == "LinearRegression":
        return LinearRegression(**model_params)
    if model_name == "ElasticNet":
        return ElasticNet(**model_params)
    if model_name == "RandomForest":
        return RandomForestRegressor(**model_params)
    if model_name == "XGBoost":
        return XGBRegressor(**model_params)
    if model_name == "CatBoost":
        return CatBoostRegressor(**model_params)
    raise ValueError(f"Unsupported ML model: {model_name}")


def impute_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    validation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Заполняет пропуски в обучающих и тестовых признаках."""
    # Медианы считаются только по train, затем без переобучения применяются к validation/test.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = pd.DataFrame(imputer.fit_transform(train[features]), columns=features, index=train.index)
    test_imputed = pd.DataFrame(imputer.transform(test[features]), columns=features, index=test.index)
    val_imputed = None
    if validation is not None:
        val_imputed = pd.DataFrame(imputer.transform(validation[features]), columns=features, index=validation.index)
    return train_imputed, val_imputed, test_imputed


def feature_importance_rows(
    estimator: object,
    features: list[str],
    *,
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    model_name: str,
) -> list[dict[str, Any]]:
    """Формирует строки важности признаков для обученной модели."""
    values = None
    kind = None
    if hasattr(estimator, "feature_importances_"):
        values = getattr(estimator, "feature_importances_")
        kind = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.ravel(getattr(estimator, "coef_"))
        kind = "coefficient"
    if values is None:
        return []
    rows = []
    for feature, value in zip(features, values):
        rows.append(
            {
                "station": station,
                "horizon": horizon,
                "horizon_hours": horizon_to_hours(horizon),
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "feature": feature,
                "importance": float(value),
                "importance_abs": float(abs(value)),
                "importance_kind": kind,
            }
        )
    return rows


def shap_enabled_for(config: dict[str, Any], model_name: str, train_days: int | None) -> bool:
    """Проверяет, нужно ли считать SHAP для модели."""
    shap_config = config.get("shap", {})
    if not shap_config.get("enabled", False):
        return False
    models = shap_config.get("models", [])
    base_model_name = model_name.removeprefix("Optuna_")
    if models and model_name not in models and base_model_name not in models:
        return False
    configured_train_days = shap_config.get("train_days", [])
    if configured_train_days and train_days not in configured_train_days:
        return False
    return True


def shap_summary_rows(
    estimator: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    config: dict[str, Any],
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    model_name: str,
) -> list[dict[str, Any]]:
    """Считает SHAP-важность признаков для модели."""
    if not shap_enabled_for(config, model_name, train_days) or not features:
        return []

    try:
        import shap
    except ImportError:
        return []

    shap_config = config.get("shap", {})
    background_rows = int(shap_config.get("sample_background_rows", 1000))
    test_rows = int(shap_config.get("sample_test_rows", 500))
    random_state = int(shap_config.get("random_state", config.get("automl", {}).get("random_state", 42)))

    y_train = pd.to_numeric(train[f"{config['target']}_target_{safe_name(horizon)}"], errors="coerce")
    valid_train = y_train.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]):
        return []

    train_imputed, _, test_imputed = impute_frames(train.loc[valid_train], test, features)
    if train_imputed.empty or test_imputed.empty:
        return []

    background = train_imputed.sample(min(background_rows, len(train_imputed)), random_state=random_state)
    sample = test_imputed.sample(min(test_rows, len(test_imputed)), random_state=random_state)
    if sample.empty:
        return []

    try:
        explainer = shap.Explainer(estimator, background)
        values = explainer(sample).values
    except Exception as exc:
        return [
            {
                "station": station,
                "horizon": horizon,
                "horizon_hours": horizon_to_hours(horizon),
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "feature": "__shap_error__",
                "mean_abs_shap": math.nan,
                "mean_shap": math.nan,
                "sample_rows": int(len(sample)),
                "background_rows": int(len(background)),
                "error": str(exc),
            }
        ]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    mean_abs = np.nanmean(np.abs(values), axis=0)
    mean = np.nanmean(values, axis=0)
    rows = []
    for feature, mean_abs_value, mean_value in zip(features, mean_abs, mean):
        rows.append(
            {
                "station": station,
                "horizon": horizon,
                "horizon_hours": horizon_to_hours(horizon),
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "feature": feature,
                "mean_abs_shap": float(mean_abs_value),
                "mean_shap": float(mean_value),
                "sample_rows": int(len(sample)),
                "background_rows": int(len(background)),
                "error": "",
            }
        )
    return rows


def fit_predict_ml(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
) -> tuple[pd.Series, int, object | None]:
    """Обучает модель и получает прогноз на тестовой выборке."""
    if not selected_features:
        return pd.Series(index=test.index, dtype=float), 0, None

    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_train = y_train.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]):
        return pd.Series(index=test.index, dtype=float), int(valid_train.sum()), None

    train_imputed, _, test_imputed = impute_frames(train.loc[valid_train], test, selected_features)
    estimator = make_estimator(model_name, config)
    estimator.fit(train_imputed, y_train.loc[valid_train])
    return pd.Series(estimator.predict(test_imputed), index=test.index), int(valid_train.sum()), estimator


def objective_direction(metric: str) -> str:
    """Определяет направление оптимизации метрики."""
    return "maximize" if metric.upper() in {"R2", "CORR", "CC"} else "minimize"


def objective_value(metric: str, y_true: pd.Series, y_pred: pd.Series) -> float:
    """Возвращает значение целевой метрики для подбора параметров."""
    metrics = metric_record(y_true, y_pred)
    key = {"RMSE": "rmse", "MAE": "mae", "R2": "r2", "CORR": "corr", "CC": "corr"}.get(metric.upper())
    if key is None:
        raise ValueError(f"Unsupported Optuna metric: {metric!r}.")
    value = metrics[key]
    if not np.isfinite(value):
        return np.inf if objective_direction(metric) == "minimize" else -np.inf
    return float(value)


def suggest_from_space(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Предлагает значение гиперпараметра из заданного пространства."""
    param_type = spec["type"]
    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if param_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if param_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported hyperparameter type for {name}: {param_type!r}")


def configured_optuna_params(model_name: str, trial: optuna.Trial, config: dict[str, Any]) -> dict[str, Any]:
    """Получает настройки Optuna из конфигурации эксперимента."""
    spaces = config.get("hyperparameter_spaces", {})
    if model_name not in spaces:
        return {}
    return {name: suggest_from_space(trial, name, spec) for name, spec in spaces[model_name].items()}


def grid_values_from_space(spec: dict[str, Any]) -> list[Any]:
    """Преобразует описание пространства параметров в значения сетки."""
    if "values" in spec:
        return list(spec["values"])
    param_type = spec["type"]
    if param_type == "categorical":
        return list(spec["choices"])
    if param_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        values = list(range(low, high + 1, step))
        if len(values) <= 5:
            return values
        return sorted({values[0], values[len(values) // 2], values[-1]})
    if param_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if bool(spec.get("log", False)) and low > 0:
            mid = float(np.sqrt(low * high))
        else:
            mid = (low + high) / 2
        return [low, mid, high]
    raise ValueError(f"Unsupported hyperparameter type: {param_type!r}")


def parameter_grid_from_config(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    """Формирует сетку параметров из конфигурации."""
    spaces = config.get("hyperparameter_spaces", {})
    if model_name not in spaces:
        return {}
    return {name: grid_values_from_space(spec) for name, spec in spaces[model_name].items()}


def fixed_model_params(model_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Возвращает фиксированные параметры модели."""
    config = config or {}
    runtime = config.get("runtime", {})
    n_jobs = int(runtime.get("model_n_jobs", 1))
    thread_count = int(runtime.get("catboost_thread_count", n_jobs))
    if model_name == "ElasticNet":
        return {"max_iter": 10000, "random_state": 42}
    if model_name == "RandomForest":
        return {"n_jobs": n_jobs, "random_state": 42}
    if model_name == "XGBoost":
        return {"objective": "reg:squarederror", "n_jobs": n_jobs, "verbosity": 0, "random_state": 42}
    if model_name == "CatBoost":
        return {"random_seed": 42, "thread_count": thread_count, "verbose": False}
    return {}


def sample_optuna_params(model_name: str, trial: optuna.Trial, config: dict[str, Any]) -> dict[str, Any]:
    """Выбирает параметры модели для одного Optuna trial."""
    configured = configured_optuna_params(model_name, trial, config)
    if model_name in OPTUNA_MODELS:
        return configured | fixed_model_params(model_name, config)
    raise ValueError(f"Unsupported Optuna model: {model_name}")


def candidate_search_params(model_name: str, config: dict[str, Any], method: str, random_state: int) -> list[dict[str, Any]]:
    """Формирует кандидаты параметров для поиска без Optuna."""
    grid = parameter_grid_from_config(model_name, config)
    if not grid:
        return [fixed_model_params(model_name, config)]
    if method == "grid_search":
        return [dict(params) | fixed_model_params(model_name, config) for params in ParameterGrid(grid)]
    if method == "random_search":
        n_iter = int(config.get("automl", {}).get("n_trials", 25))
        return [dict(params) | fixed_model_params(model_name, config) for params in ParameterSampler(grid, n_iter=n_iter, random_state=random_state)]
    if method == "default":
        return [default_model_params(model_name, config)]
    raise ValueError(f"Unsupported search method: {method!r}")


def fit_final_tuned_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.Series, object, int]:
    """Обучает финальную модель с выбранными параметрами."""
    automl = config.get("automl", {})
    final_train = train
    if automl.get("train_final_on_train_validation", True):
        final_train = pd.concat([train, validation], axis=0)
    final_y = pd.to_numeric(final_train[target_column], errors="coerce")
    valid_final = final_y.notna()
    train_label_rows = int(valid_final.sum())
    final_x, _, final_test_x = impute_frames(final_train.loc[valid_final], test, selected_features)
    final_model = make_estimator(model_name, config, params=params)
    final_model.fit(final_x, final_y.loc[valid_final])
    y_pred = pd.Series(final_model.predict(final_test_x), index=test.index)
    return y_pred, final_model, train_label_rows


def run_candidate_search_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    output_dir: Path,
    method: str,
) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, object | None]:
    """Запускает подбор параметров перебором кандидатов."""
    automl = config.get("automl", {})
    metric = automl.get("metric", "RMSE")
    random_state = int(automl.get("random_state", 42))
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    y_val = pd.to_numeric(validation[target_column], errors="coerce")
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    valid_train = y_train.notna()
    valid_val = y_val.notna()
    valid_test = y_test.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]) or valid_val.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), {}, [], [], [], int(valid_train.sum()), None

    x_train, x_val, _ = impute_frames(train.loc[valid_train], test, selected_features, validation.loc[valid_val])
    assert x_val is not None
    y_train_fit = y_train.loc[valid_train]
    y_val_eval = y_val.loc[valid_val]
    candidates = candidate_search_params(model_name, config, method, random_state)
    direction = objective_direction(metric)
    trial_rows = []
    best_value = np.inf if direction == "minimize" else -np.inf
    best_params: dict[str, Any] = {}

    for trial_number, params in enumerate(candidates):
        model = make_estimator(model_name, config, params=params)
        model.fit(x_train, y_train_fit)
        pred = pd.Series(model.predict(x_val), index=y_val_eval.index)
        value = objective_value(metric, y_val_eval, pred)
        is_better = value < best_value if direction == "minimize" else value > best_value
        if is_better:
            best_value = value
            best_params = dict(params)
        trial_rows.append(
            {
                "station": station,
                "horizon": horizon,
                "horizon_hours": horizon_to_hours(horizon),
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "study_name": f"{station}_{safe_name(horizon)}_{model_name}_{target_column}_{split_id}_{method}",
                "trial_number": trial_number,
                "value": float(value),
                "state": "COMPLETE",
                "params": json.dumps(params, sort_keys=True),
            }
        )

    if not best_params:
        return pd.Series(index=test.index, dtype=float), {}, trial_rows, [], [], int(valid_train.sum()), None

    y_pred, final_model, train_label_rows = fit_final_tuned_model(
        train,
        validation,
        test,
        target_column=target_column,
        selected_features=selected_features,
        model_name=model_name,
        config=config,
        params=best_params,
    )
    model_key = f"{station}_{safe_name(horizon)}_{model_name}_{split_id}_{method}"
    if automl.get("save_models", False):
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_dir / f"{model_key}.joblib")
    info = {
        "best_params": json.dumps(best_params, sort_keys=True),
        "best_val_score": float(best_value),
        "study_name": model_key,
        "search_method": method,
    }
    best_param_rows = [
        {
            "station": station,
            "horizon": horizon,
            "horizon_hours": horizon_to_hours(horizon),
            "split_id": split_id,
            "train_days": train_days,
            "model": model_name,
            "study_name": model_key,
            "search_method": method,
            "best_value": float(best_value),
            "best_params": json.dumps(best_params, sort_keys=True),
            "train_final_rows": train_label_rows,
        }
    ]
    importance_rows = feature_importance_rows(
        final_model,
        selected_features,
        station=station,
        horizon=horizon,
        split_id=split_id,
        train_days=train_days,
        model_name=f"{method}_{model_name}",
    )
    if valid_test.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), info, trial_rows, best_param_rows, importance_rows, train_label_rows, final_model
    return y_pred, info, trial_rows, best_param_rows, importance_rows, train_label_rows, final_model


def run_optuna_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    selected_features: list[str],
    model_name: str,
    config: dict[str, Any],
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    output_dir: Path,
) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, object | None]:
    """Запускает Optuna-подбор и обучение модели."""
    automl = config.get("automl", {})
    method = automl.get("method", "optuna_tpe_pruning")
    if method in {"default", "grid_search", "random_search"}:
        return run_candidate_search_model(
            train,
            validation,
            test,
            target_column=target_column,
            selected_features=selected_features,
            model_name=model_name,
            config=config,
            station=station,
            horizon=horizon,
            split_id=split_id,
            train_days=train_days,
            output_dir=output_dir,
            method=method,
        )
    metric = automl.get("metric", "RMSE")
    random_state = int(automl.get("random_state", 42))
    direction = objective_direction(metric)

    y_train = pd.to_numeric(train[target_column], errors="coerce")
    y_val = pd.to_numeric(validation[target_column], errors="coerce")
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    valid_train = y_train.notna()
    valid_val = y_val.notna()
    valid_test = y_test.notna()
    if valid_train.sum() < int(config["minimum_train_rows"]) or valid_val.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), {}, [], [], [], int(valid_train.sum()), None

    # Optuna подбирает параметры на validation, а test остается отложенным набором для финальной оценки.
    x_train, x_val, x_test = impute_frames(train.loc[valid_train], test, selected_features, validation.loc[valid_val])
    assert x_val is not None
    y_train_fit = y_train.loc[valid_train]
    y_val_eval = y_val.loc[valid_val]

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner() if method == "optuna_tpe_pruning" else optuna.pruners.NopPruner()
    storage = automl.get("storage")
    if isinstance(storage, str):
        storage = storage.format(station=station, horizon=safe_name(horizon), split_id=split_id, model=model_name)
        if storage.startswith("sqlite:///"):
            Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    study_name = f"{station}_{safe_name(horizon)}_{model_name}_{target_column}_{split_id}"
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        """Выполняет вспомогательный шаг objective в текущем сценарии."""
        params = sample_optuna_params(model_name, trial, config)
        model = make_estimator(model_name, config, params=params)
        model.fit(x_train, y_train_fit)
        pred = pd.Series(model.predict(x_val), index=y_val_eval.index)
        value = objective_value(metric, y_val_eval, pred)
        trial.report(value, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value

    target_trials = int(automl.get("n_trials", 25))
    remaining_trials = max(0, target_trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    best_params = dict(study.best_params)
    best_value = float(study.best_value)

    final_train = train
    if automl.get("train_final_on_train_validation", True):
        final_train = pd.concat([train, validation], axis=0)
    final_y = pd.to_numeric(final_train[target_column], errors="coerce")
    valid_final = final_y.notna()
    train_label_rows = int(valid_final.sum())
    final_x, _, final_test_x = impute_frames(final_train.loc[valid_final], test, selected_features)
    final_model = make_estimator(model_name, config, params=best_params)
    final_model.fit(final_x, final_y.loc[valid_final])
    y_pred = pd.Series(final_model.predict(final_test_x), index=test.index)

    if automl.get("save_models", False):
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_dir / f"{station}_{safe_name(horizon)}_{model_name}_{split_id}.joblib")

    trials = []
    for trial in study.trials:
        trials.append(
            {
                "station": station,
                "horizon": horizon,
                "horizon_hours": horizon_to_hours(horizon),
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "study_name": study_name,
                "trial_number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                "params": json.dumps(trial.params, sort_keys=True),
            }
        )
    info = {"best_params": json.dumps(best_params, sort_keys=True), "best_val_score": best_value, "study_name": study_name}
    best_param_rows = [
        {
            "station": station,
            "horizon": horizon,
            "horizon_hours": horizon_to_hours(horizon),
            "split_id": split_id,
            "train_days": train_days,
            "model": model_name,
            "study_name": study_name,
            "best_value": best_value,
            "best_params": json.dumps(best_params, sort_keys=True),
            "train_final_rows": train_label_rows,
        }
    ]
    importance_rows = feature_importance_rows(
        final_model,
        selected_features,
        station=station,
        horizon=horizon,
        split_id=split_id,
        train_days=train_days,
        model_name=f"Optuna_{model_name}",
    )
    if valid_test.sum() < int(config["minimum_eval_rows"]):
        return pd.Series(index=test.index, dtype=float), info, trials, best_param_rows, importance_rows, train_label_rows, final_model
    return y_pred, info, trials, best_param_rows, importance_rows, train_label_rows, final_model


def append_metric_and_predictions(
    *,
    metric_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
    station: str,
    horizon: str,
    split_id: str,
    train_days: int | None,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    selected_features: list[str],
    target_column: str,
    y_pred: pd.Series,
    extra: dict[str, Any] | None = None,
) -> None:
    """Добавляет метрики и прогнозы одной итерации в накопители."""
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    metrics = metric_record(y_test, y_pred)
    if metrics["n"] <= 0:
        return
    extra = extra or {}
    metric_rows.append(
        {
            "station": station,
            "horizon": horizon,
            "horizon_hours": horizon_to_hours(horizon),
            "split_id": split_id,
            "train_days": train_days,
            "model": model_name,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "feature_count": int(len(selected_features)),
            **metrics,
            **extra,
            **{key: value.isoformat() for key, value in bounds.items()},
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "station": station,
                "time_utc": test["time_utc"],
                "target_time_utc": test["time_utc"] + pd.Timedelta(horizon),
                "horizon": horizon,
                "split_id": split_id,
                "train_days": train_days,
                "model": model_name,
                "actual": y_test,
                "predicted": y_pred,
            }
        )
    )


def run_station(
    config: dict[str, Any],
    station: str,
    horizons: list[str],
    output_dir: Path,
    run_automl: bool,
) -> tuple[
    list[dict[str, Any]],
    list[pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Запускает полный ML-эксперимент для одной станции."""
    path = Path(config["feature_dir"]) / "stations" / f"{station}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    feature_pool = numeric_feature_pool(frame, config["target"])
    context = station_context(station, frame)
    station_dir = output_dir / "stations" / station
    station_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []
    best_param_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []
    iterations = split_iterations(config)
    total_splits = len(iterations)
    for horizon in horizons:
        target_column = f"{config['target']}_target_{safe_name(horizon)}"
        if target_column not in frame.columns:
            raise KeyError(f"Missing target column {target_column!r} in {path}.")
        for split_index, iteration in enumerate(iterations, start=1):
            start_metrics = len(metric_rows)
            start_predictions = len(prediction_frames)
            start_trials = len(trial_rows)
            start_best_params = len(best_param_rows)
            start_importance = len(importance_rows)
            start_shap = len(shap_rows)
            train, validation, test, bounds = split_frames(
                frame,
                config=config,
                horizon=horizon,
                current_day=iteration["test_day"],
                train_days=iteration["train_days"],
            )
            # После разбиения признаки выбираются заново: покрытие меняется от окна к окну.
            selected_features = select_features(train, feature_pool, float(config["minimum_feature_coverage"]))
            y_test = pd.to_numeric(test[target_column], errors="coerce")
            if (
                len(train) < int(config["minimum_train_rows"])
                or y_test.notna().sum() < int(config["minimum_eval_rows"])
                or not selected_features
            ):
                continue

            for model_name in config["models"]:
                print(
                    f"[{station}] horizon={horizon} window={iteration['train_days']}d "
                    f"split={split_index}/{total_splits} model={model_name}",
                    flush=True,
                )
                if model_name in BASELINE_MODELS:
                    y_pred = make_baseline_predictions(test, model_name, config["target"], horizon)
                    extra = {"train_label_rows": int(pd.to_numeric(train[target_column], errors="coerce").notna().sum())}
                elif model_name in ML_MODELS:
                    y_pred, train_label_rows, estimator = fit_predict_ml(
                        train,
                        test,
                        target_column=target_column,
                        selected_features=selected_features,
                        model_name=model_name,
                        config=config,
                    )
                    extra = {"train_label_rows": train_label_rows}
                    if estimator is not None:
                        importance_rows.extend(
                            feature_importance_rows(
                                estimator,
                                selected_features,
                                station=station,
                                horizon=horizon,
                                split_id=iteration["split_id"],
                                train_days=iteration["train_days"],
                                model_name=model_name,
                            )
                        )
                        shap_rows.extend(
                            shap_summary_rows(
                                estimator,
                                train,
                                test,
                                selected_features,
                                config=config,
                                station=station,
                                horizon=horizon,
                                split_id=iteration["split_id"],
                                train_days=iteration["train_days"],
                                model_name=model_name,
                            )
                        )
                else:
                    raise ValueError(f"Unsupported model in config: {model_name}")
                append_metric_and_predictions(
                    metric_rows=metric_rows,
                    prediction_frames=prediction_frames,
                    station=station,
                    horizon=horizon,
                    split_id=iteration["split_id"],
                    train_days=iteration["train_days"],
                    model_name=model_name,
                    train=train,
                    validation=validation,
                    test=test,
                    bounds=bounds,
                    selected_features=selected_features,
                    target_column=target_column,
                    y_pred=y_pred,
                    extra=extra,
                )

            automl = config.get("automl", {})
            if run_automl and automl.get("enabled", False):
                for model_name in automl.get("models", []):
                    if model_name not in OPTUNA_MODELS:
                        raise ValueError(f"Unsupported Optuna model in config: {model_name}")
                    y_pred, info, trials, best_params, optuna_importance, train_label_rows, estimator = run_optuna_model(
                        train,
                        validation,
                        test,
                        target_column=target_column,
                        selected_features=selected_features,
                        model_name=model_name,
                        config=config,
                        station=station,
                        horizon=horizon,
                        split_id=iteration["split_id"],
                        train_days=iteration["train_days"],
                        output_dir=output_dir,
                    )
                    trial_rows.extend(trials)
                    best_param_rows.extend(best_params)
                    importance_rows.extend(optuna_importance)
                    if estimator is not None:
                        shap_rows.extend(
                            shap_summary_rows(
                                estimator,
                                pd.concat([train, validation], axis=0)
                                if automl.get("train_final_on_train_validation", True)
                                else train,
                                test,
                                selected_features,
                                config=config,
                                station=station,
                                horizon=horizon,
                                split_id=iteration["split_id"],
                                train_days=iteration["train_days"],
                                model_name=f"Optuna_{model_name}",
                            )
                        )
                    append_metric_and_predictions(
                        metric_rows=metric_rows,
                        prediction_frames=prediction_frames,
                        station=station,
                        horizon=horizon,
                        split_id=iteration["split_id"],
                        train_days=iteration["train_days"],
                        model_name=f"Optuna_{model_name}",
                        train=train,
                        validation=validation,
                        test=test,
                        bounds=bounds,
                        selected_features=selected_features,
                        target_column=target_column,
                        y_pred=y_pred,
                        extra={"train_label_rows": train_label_rows, **info},
                    )
            new_metrics = enrich_rows(metric_rows[start_metrics:], context)
            new_predictions = prediction_frames[start_predictions:]
            new_trials = enrich_rows(trial_rows[start_trials:], context)
            new_best_params = enrich_rows(best_param_rows[start_best_params:], context)
            new_importance = enrich_rows(importance_rows[start_importance:], context)
            new_shap = enrich_rows(shap_rows[start_shap:], context)
            write_partial_station_outputs(
                station_dir,
                new_metrics,
                new_predictions,
                new_trials,
                new_best_params,
                new_importance,
                new_shap,
                station=station,
                horizon=horizon,
                split_id=iteration["split_id"],
            )
    metric_rows = enrich_rows(metric_rows, context)
    trial_rows = enrich_rows(trial_rows, context)
    best_param_rows = enrich_rows(best_param_rows, context)
    importance_rows = enrich_rows(importance_rows, context)
    shap_rows = enrich_rows(shap_rows, context)
    return metric_rows, prediction_frames, trial_rows, best_param_rows, importance_rows, shap_rows


def append_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Добавляет строки в CSV-файл."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_prediction_frames(frames: list[pd.DataFrame], path: Path) -> None:
    """Добавляет таблицы прогнозов в общий список."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_partial_station_outputs(
    station_dir: Path,
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    trials: list[dict[str, Any]],
    best_params: list[dict[str, Any]],
    feature_importance: list[dict[str, Any]],
    shap_summary: list[dict[str, Any]],
    *,
    station: str,
    horizon: str,
    split_id: str,
) -> None:
    """Сохраняет промежуточные результаты станции."""
    append_csv(metrics, station_dir / "partial_metrics.csv")
    append_prediction_frames(predictions, station_dir / "partial_predictions.csv")
    append_csv(trials, station_dir / "partial_optuna_trials.csv")
    append_csv(best_params, station_dir / "partial_best_params.csv")
    append_csv(feature_importance, station_dir / "partial_feature_importance.csv")
    append_csv(shap_summary, station_dir / "partial_shap_summary.csv")
    progress = {
        "station": station,
        "horizon": horizon,
        "split_id": split_id,
        "metrics_rows": sum(1 for _ in open(station_dir / "partial_metrics.csv", encoding="utf-8")) - 1
        if (station_dir / "partial_metrics.csv").exists()
        else 0,
        "updated_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    (station_dir / "partial_progress.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> None:
    """Сохраняет таблицу в CSV и Parquet при необходимости."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None and not df.empty:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)


def season_name(month: int) -> str:
    """Возвращает название сезона по номеру месяца."""
    if month in {12, 1, 2}:
        return "Зима"
    if month in {3, 4, 5}:
        return "Весна"
    if month in {6, 7, 8}:
        return "Лето"
    return "Осень"


def fmt_metric(value: Any, digits: int = 3, suffix: str = "") -> str:
    """Форматирует числовую метрику для markdown-отчета."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def model_label(model: str) -> str:
    """Возвращает читаемое название модели."""
    labels = {
        "persistence_state": "Persistence state",
        "persistence_horizon_lag": "Persistence lag",
        "seasonal_24h_lag": "Seasonal 24h lag",
    }
    return labels.get(model, model)


def available_report_metrics(df: pd.DataFrame) -> list[tuple[str, str, int, str]]:
    """Определяет метрики, доступные для отчета."""
    candidates = [
        ("r2", "R2", 3, ""),
        ("rmse", "RMSE", 2, ""),
        ("mae", "MAE", 2, ""),
        ("mape", "MAPE, %", 1, ""),
    ]
    return [(column, label, digits, suffix) for column, label, digits, suffix in candidates if column in df.columns]


def seasonal_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Собирает таблицу сезонных метрик."""
    if metrics_df.empty or "test_start" not in metrics_df.columns:
        return pd.DataFrame()
    df = metrics_df.copy()
    df["test_start"] = pd.to_datetime(df["test_start"], utc=True, errors="coerce")
    df = df.dropna(subset=["test_start"])
    if df.empty:
        return pd.DataFrame()
    df["season"] = df["test_start"].dt.month.map(season_name)
    report_metric_columns = [column for column, _, _, _ in available_report_metrics(df)]
    if not report_metric_columns:
        return pd.DataFrame()
    grouped = (
        df.groupby(["horizon", "train_days", "season", "model"], dropna=False)[["n", *report_metric_columns]]
        .mean(numeric_only=True)
        .reset_index()
    )
    order = {"Зима": 0, "Весна": 1, "Лето": 2, "Осень": 3}
    grouped["season_order"] = grouped["season"].map(order)
    return grouped.sort_values(["horizon", "train_days", "season_order", "model"]).drop(columns=["season_order"])


def markdown_metrics_table(seasonal_df: pd.DataFrame, horizon: str, train_days: Any) -> list[str]:
    """Формирует markdown-таблицу метрик."""
    subset = seasonal_df[(seasonal_df["horizon"] == horizon) & (seasonal_df["train_days"] == train_days)]
    if subset.empty:
        return []
    models = list(dict.fromkeys(subset["model"].tolist()))
    report_metrics = available_report_metrics(subset)
    lines = [
        f"### Таблица: метрики прогноза по сезонам, горизонт {horizon}, окно {int(train_days) if pd.notna(train_days) else '-'} дней",
        "",
        "| Сезон | " + " | ".join(f"{model_label(model)} {label}" for model in models for _, label, _, _ in report_metrics) + " |",
        "|---|" + "|".join("---" for _ in range(len(models) * len(report_metrics))) + "|",
    ]
    for season in ["Зима", "Весна", "Лето", "Осень"]:
        season_rows = subset[subset["season"] == season]
        if season_rows.empty:
            continue
        cells = [season]
        for model in models:
            row = season_rows[season_rows["model"] == model]
            if row.empty:
                cells.extend(["-" for _ in report_metrics])
                continue
            record = row.iloc[0]
            cells.extend([fmt_metric(record[column], digits, suffix) for column, _, digits, suffix in report_metrics])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def seasonal_analysis_lines(seasonal_df: pd.DataFrame, horizon: str, train_days: Any) -> list[str]:
    """Готовит текстовые выводы по сезонному анализу."""
    subset = seasonal_df[(seasonal_df["horizon"] == horizon) & (seasonal_df["train_days"] == train_days)]
    if subset.empty:
        return []
    lines = [f"### Краткий анализ, горизонт {horizon}, окно {int(train_days) if pd.notna(train_days) else '-'} дней", ""]
    for season in ["Зима", "Весна", "Лето", "Осень"]:
        season_rows = subset[subset["season"] == season].copy()
        season_rows = season_rows[season_rows["n"] > 0]
        if season_rows.empty:
            lines.append(f"{season}: недостаточно валидных точек для устойчивой оценки.")
            continue
        parts = []
        if "r2" in season_rows.columns:
            sort_columns = ["r2"] + (["rmse"] if "rmse" in season_rows.columns else [])
            ascending = [False] + ([True] if "rmse" in season_rows.columns else [])
            best_r2 = season_rows.sort_values(sort_columns, ascending=ascending).iloc[0]
            worst_r2 = season_rows.sort_values(sort_columns, ascending=[not value for value in ascending]).iloc[0]
            parts.append(f"лучшая модель по R2 — {model_label(best_r2['model'])} (R2={fmt_metric(best_r2['r2'])})")
            parts.append(f"наиболее слабый R2 у {model_label(worst_r2['model'])} ({fmt_metric(worst_r2['r2'])})")
        if "rmse" in season_rows.columns:
            sort_columns = ["rmse"] + (["r2"] if "r2" in season_rows.columns else [])
            ascending = [True] + ([False] if "r2" in season_rows.columns else [])
            best_rmse = season_rows.sort_values(sort_columns, ascending=ascending).iloc[0]
            parts.append(f"минимальная RMSE у {model_label(best_rmse['model'])} ({fmt_metric(best_rmse['rmse'], 2)})")
        if "mape" in season_rows.columns:
            best_mape = season_rows.sort_values("mape", ascending=True).iloc[0]
            parts.append(f"минимальная MAPE у {model_label(best_mape['model'])} ({fmt_metric(best_mape['mape'], 1, ' %')})")
        lines.append(f"{season}: " + "; ".join(parts) + ".")
    lines.append("")
    return lines


def feature_importance_analysis_lines(importance_df: pd.DataFrame) -> list[str]:
    """Готовит текстовые выводы по важности признаков."""
    if importance_df.empty or not {"model", "feature", "importance_abs"}.issubset(importance_df.columns):
        return []
    lines = ["### Важность признаков", ""]
    grouped = (
        importance_df.groupby(["model", "feature"], dropna=False)["importance_abs"]
        .mean()
        .reset_index()
        .sort_values(["model", "importance_abs"], ascending=[True, False])
    )
    for model in list(dict.fromkeys(grouped["model"].tolist())):
        top = grouped[grouped["model"] == model].head(5)
        if top.empty:
            continue
        features = ", ".join(f"{row.feature} ({fmt_metric(row.importance_abs, 3)})" for row in top.itertuples())
        lines.append(f"{model_label(model)}: ведущие признаки по средней абсолютной важности — {features}.")
    lines.append("")
    return lines


def write_station_markdown_report(
    station_dir: Path,
    config: dict[str, Any],
    station: str,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Создает markdown-отчет по отдельной станции."""
    seasonal_df = seasonal_metrics_table(metrics_df)
    lines = [
        f"# Отчет по станции {station}",
        "",
        f"- Эксперимент: `{config.get('experiment_id', '')}`",
        f"- Целевой параметр: `{config.get('target', '')}`",
        f"- Время запуска UTC: `{config.get('run_started_utc', '')}`",
        f"- Строк метрик: {len(metrics_df)}",
        "",
    ]
    if seasonal_df.empty:
        lines.extend(["Недостаточно данных для сезонной агрегации метрик.", ""])
    else:
        for horizon in list(dict.fromkeys(seasonal_df["horizon"].tolist())):
            horizon_df = seasonal_df[seasonal_df["horizon"] == horizon]
            for train_days in list(dict.fromkeys(horizon_df["train_days"].tolist())):
                lines.extend(markdown_metrics_table(seasonal_df, horizon, train_days))
                lines.extend(seasonal_analysis_lines(seasonal_df, horizon, train_days))
    lines.extend(feature_importance_analysis_lines(importance_df))
    lines.extend(
        [
            "### Примечание",
            "",
            "В отчет включаются только метрики, которые есть в таблице результатов. Сезоны определяются по дате начала тестового интервала.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_station_outputs(
    output_dir: Path,
    config: dict[str, Any],
    station: str,
    metrics: list[dict[str, Any]],
    predictions: pd.DataFrame,
    trials: list[dict[str, Any]],
    best_params: list[dict[str, Any]],
    feature_importance: list[dict[str, Any]],
    shap_summary: list[dict[str, Any]],
) -> dict[str, int]:
    """Сохраняет все результаты эксперимента по станции."""
    station_dir = output_dir / "stations" / station
    station_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(config, station)
    metrics_df = pd.DataFrame(metrics)
    trials_df = pd.DataFrame(trials)
    best_params_df = pd.DataFrame(best_params)
    importance_df = pd.DataFrame(feature_importance)
    shap_df = pd.DataFrame(shap_summary)

    files = {
        "metrics_csv": f"{prefix}_metrics.csv",
        "metrics_parquet": f"{prefix}_metrics.parquet",
        "predictions_csv": f"{prefix}_predictions.csv",
        "predictions_parquet": f"{prefix}_predictions.parquet",
        "optuna_trials_csv": f"{prefix}_optuna_trials.csv",
        "optuna_trials_parquet": f"{prefix}_optuna_trials.parquet",
        "best_params_csv": f"{prefix}_best_params.csv",
        "best_params_parquet": f"{prefix}_best_params.parquet",
        "feature_importance_csv": f"{prefix}_feature_importance.csv",
        "feature_importance_parquet": f"{prefix}_feature_importance.parquet",
        "shap_summary_csv": f"{prefix}_shap_summary.csv",
        "shap_summary_parquet": f"{prefix}_shap_summary.parquet",
        "station_report": f"{prefix}_report.md",
        "run_summary": f"{prefix}_run_summary.json",
    }

    write_table(metrics_df, station_dir / files["metrics_csv"], station_dir / files["metrics_parquet"])
    write_table(predictions, station_dir / files["predictions_csv"], station_dir / files["predictions_parquet"])
    write_table(trials_df, station_dir / files["optuna_trials_csv"], station_dir / files["optuna_trials_parquet"])
    write_table(best_params_df, station_dir / files["best_params_csv"], station_dir / files["best_params_parquet"])
    write_table(importance_df, station_dir / files["feature_importance_csv"], station_dir / files["feature_importance_parquet"])
    write_table(shap_df, station_dir / files["shap_summary_csv"], station_dir / files["shap_summary_parquet"])
    write_station_markdown_report(station_dir, config, station, metrics_df, importance_df, station_dir / files["station_report"])

    station_summary = {
        "station": station,
        "experiment_id": config.get("experiment_id"),
        "run_started_utc": config.get("run_started_utc"),
        "file_prefix": prefix,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(len(predictions)),
        "trial_rows": int(len(trials_df)),
        "best_param_rows": int(len(best_params_df)),
        "feature_importance_rows": int(len(importance_df)),
        "shap_summary_rows": int(len(shap_df)),
        "files": files,
    }
    (station_dir / files["run_summary"]).write_text(json.dumps(station_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: int(value) for key, value in station_summary.items() if key.endswith("_rows")}


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    stations: list[str],
    horizons: list[str],
    metrics: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    best_params: list[dict[str, Any]],
    feature_importance: list[dict[str, Any]],
    shap_summary: list[dict[str, Any]],
    prediction_rows: int,
    station_summaries: list[dict[str, Any]],
) -> None:
    """Сохраняет итоговые результаты всего эксперимента."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    trials_df = pd.DataFrame(trials)
    best_params_df = pd.DataFrame(best_params)
    importance_df = pd.DataFrame(feature_importance)
    shap_df = pd.DataFrame(shap_summary)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    best_params_df.to_csv(output_dir / "best_params.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    shap_df.to_csv(output_dir / "shap_summary.csv", index=False)
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["station", "horizon", "model"], dropna=False)[["n", "mae", "rmse", "r2", "corr", "bias"]]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    if config.get("automl", {}).get("export_trials_parquet", False) and not trials_df.empty:
        trials_df.to_parquet(output_dir / "optuna_trials.parquet", index=False)
    if not best_params_df.empty:
        best_params_df.to_parquet(output_dir / "best_params.parquet", index=False)
    if not importance_df.empty:
        importance_df.to_parquet(output_dir / "feature_importance.parquet", index=False)
    if not shap_df.empty:
        shap_df.to_parquet(output_dir / "shap_summary.parquet", index=False)
    if not metrics_df.empty:
        metrics_df.to_parquet(output_dir / "metrics.parquet", index=False)
    pd.DataFrame(station_summaries).to_csv(output_dir / "station_run_summary.csv", index=False)
    registry_columns = [
        "station",
        "latitude_zone",
        "horizon",
        "split_id",
        "train_days",
        "model",
        "train_start",
        "safe_train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "n",
        "mae",
        "rmse",
        "r2",
        "corr",
        "bias",
        "best_val_score",
        "study_name",
    ]
    registry_columns = [column for column in registry_columns if column in metrics_df.columns]
    model_registry = metrics_df[registry_columns].copy() if registry_columns else pd.DataFrame()
    model_registry.to_csv(output_dir / "model_registry.csv", index=False)
    if not model_registry.empty:
        model_registry.to_parquet(output_dir / "model_registry.parquet", index=False)

    run_manifest = {
        "config": config,
        "stations": stations,
        "horizons": horizons,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(prediction_rows),
        "trial_rows": int(len(trials_df)),
        "best_param_rows": int(len(best_params_df)),
        "feature_importance_rows": int(len(importance_df)),
        "shap_summary_rows": int(len(shap_df)),
        "station_summaries": station_summaries,
        "result_layout": {
            "global": [
                "metrics.csv",
                "metrics_summary.csv",
                "optuna_trials.csv",
                "best_params.csv",
                "feature_importance.csv",
                "shap_summary.csv",
                "model_registry.csv",
                "experiment_report.md",
                "run_manifest.json",
            ],
            "per_station": "artifacts/<experiment>/stations/<station>/<station>_<experiment>_<run_started_utc>_<table>.{csv,parquet}",
            "models": "artifacts/<experiment>/models/<station>_<horizon>_<model>_<split_id>.joblib",
            "optuna_storage": config.get("automl", {}).get("storage"),
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_lines = [
        f"# Experiment {config.get('experiment_id', '')}",
        "",
        f"- dataset_step: {config.get('dataset_step')}",
        f"- evaluation_mode: {config.get('evaluation_mode')}",
        f"- stations: {len(stations)}",
        f"- horizons: {', '.join(horizons)}",
        f"- metrics_rows: {len(metrics_df)}",
        f"- prediction_rows: {prediction_rows}",
        f"- search_trial_rows: {len(trials_df)}",
        f"- best_param_rows: {len(best_params_df)}",
        f"- feature_importance_rows: {len(importance_df)}",
        "",
        "## Output Tables",
        "",
        "- metrics.csv / metrics.parquet",
        "- metrics_summary.csv",
        "- station_run_summary.csv",
        "- model_registry.csv / model_registry.parquet",
        "- optuna_trials.csv / optuna_trials.parquet",
        "- best_params.csv / best_params.parquet",
        "- feature_importance.csv / feature_importance.parquet",
        "- stations/<station>/<station>_<experiment>_<run_started_utc>_<table>.*",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def dry_run(config: dict[str, Any], stations: list[str], horizons: list[str], run_automl: bool) -> None:
    """Проверяет настройки запуска без обучения моделей."""
    iterations = split_iterations(config)
    regular_models = len(config.get("models", []))
    optuna_models = len(config.get("automl", {}).get("models", [])) if run_automl and config.get("automl", {}).get("enabled") else 0
    total_fit_groups = len(stations) * len(horizons) * len(iterations)
    total_regular_model_fits = total_fit_groups * regular_models
    total_optuna_trials = total_fit_groups * optuna_models * int(config.get("automl", {}).get("n_trials", 0))
    print("DRY RUN")
    print(f"evaluation_mode: {config.get('evaluation_mode')}")
    print(f"stations: {len(stations)}")
    print(f"horizons: {horizons}")
    print(f"split_iterations: {len(iterations)}")
    print(f"regular_models: {regular_models} -> estimated fits {total_regular_model_fits}")
    print(f"optuna_models: {optuna_models} -> estimated trials {total_optuna_trials}")
    print(f"output_dir: {config['output_dir']}")
    if config.get("evaluation_mode") == "fixed_year_split" and stations and horizons:
        sample_path = Path(config["feature_dir"]) / "stations" / f"{stations[0]}_features.csv"
        sample = pd.read_csv(sample_path, usecols=["time_utc", f"{config['target']}_target_{safe_name(horizons[-1])}"])
        sample["time_utc"] = pd.to_datetime(sample["time_utc"], utc=True, errors="coerce")
        train, validation, test, bounds = split_fixed_year(
            sample,
            forecast_horizon=horizons[-1],
            dataset_step=config["dataset_step"],
            config=config,
        )
        print("sample_split_for_largest_horizon:")
        for key, value in bounds.items():
            print(f"  {key}: {value}")
        print(f"  rows: train={len(train)} validation={len(validation)} test={len(test)}")
    elif config.get("evaluation_mode") == "walk_forward_daily" and stations and horizons:
        sample_path = Path(config["feature_dir"]) / "stations" / f"{stations[0]}_features.csv"
        sample = pd.read_csv(sample_path, usecols=["time_utc", f"{config['target']}_target_{safe_name(horizons[-1])}"])
        sample["time_utc"] = pd.to_datetime(sample["time_utc"], utc=True, errors="coerce")
        first_iteration = iterations[0]
        train, validation, test, bounds = split_frames(
            sample,
            config=config,
            horizon=horizons[-1],
            current_day=first_iteration["test_day"],
            train_days=first_iteration["train_days"],
        )
        print("sample_first_walk_forward_split_for_largest_horizon:")
        for key, value in bounds.items():
            print(f"  {key}: {value}")
        print(f"  rows: train={len(train)} validation={len(validation)} test={len(test)}")


def main() -> None:
    """Запускает основной сценарий файла."""
    args = parse_args()
    config = load_config(Path(args.config))
    config.setdefault("run_started_utc", utc_run_stamp())
    stations = resolve_stations(config, args.station)
    horizons = [args.horizon] if args.horizon else list(config["horizons"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_automl = not args.no_automl
    if args.dry_run:
        dry_run(config, stations, horizons, run_automl)
        return

    all_metrics: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    all_best_params: list[dict[str, Any]] = []
    all_importance: list[dict[str, Any]] = []
    all_shap: list[dict[str, Any]] = []
    station_summaries: list[dict[str, Any]] = []
    prediction_rows = 0
    for station in stations:
        metrics, predictions, trials, best_params, importance, shap_summary = run_station(
            config, station, horizons, output_dir, run_automl
        )
        all_metrics.extend(metrics)
        all_trials.extend(trials)
        all_best_params.extend(best_params)
        all_importance.extend(importance)
        all_shap.extend(shap_summary)
        station_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        prediction_rows += int(len(station_predictions))
        summary_counts = write_station_outputs(
            output_dir, config, station, metrics, station_predictions, trials, best_params, importance, shap_summary
        )
        station_summaries.append({"station": station, **summary_counts})
        print(
            f"{station}: metrics={len(metrics)} prediction_blocks={len(predictions)} "
            f"trials={len(trials)} best_params={len(best_params)} "
            f"feature_importance={len(importance)} shap_summary={len(shap_summary)}"
        )

    write_outputs(
        output_dir,
        config,
        stations,
        horizons,
        all_metrics,
        all_trials,
        all_best_params,
        all_importance,
        all_shap,
        prediction_rows,
        station_summaries,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
