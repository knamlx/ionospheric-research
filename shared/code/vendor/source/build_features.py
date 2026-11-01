from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска из командной строки."""
    parser = argparse.ArgumentParser(description="Build leak-aware ML feature tables from station time-grid CSVs.")
    parser.add_argument("--input-dir", default="normalized_2024_2025_exploration_v0_1_15min_by_station")
    parser.add_argument("--config", default="configs/feature_engineering.json")
    parser.add_argument("--output-dir", default="features_2024_2025_exploration_v0_1_15min")
    parser.add_argument("--station", default=None, help="Optional single station code for a quick run.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Загружает конфигурацию эксперимента или сбора данных из файла."""
    return json.loads(path.read_text(encoding="utf-8"))


def duration_to_steps(duration: str, dataset_step: str) -> int:
    """Переводит длительность временного лага в число шагов сетки."""
    delta = pd.Timedelta(duration)
    step = pd.Timedelta(dataset_step)
    if delta < step:
        raise ValueError(f"Duration {duration!r} is shorter than dataset_step {dataset_step!r}.")
    if delta % step != pd.Timedelta(0):
        raise ValueError(f"Duration {duration!r} is not divisible by dataset_step {dataset_step!r}.")
    return int(delta / step)


def safe_name(duration: str) -> str:
    """Преобразует строку в безопасное имя для файлов и папок."""
    return duration.replace(" ", "").replace("/", "_").replace("-", "_")


def add_time_features(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Добавляет календарные и циклические временные признаки."""
    time = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame["hour"] = time.dt.hour
    frame["month"] = time.dt.month
    frame["doy"] = time.dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["doy"] / 366)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["doy"] / 366)
    return frame


def add_lag_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Добавляет лаговые признаки для выбранных параметров."""
    dataset_step = config["dataset_step"]
    additions: dict[str, pd.Series] = {}
    # Текущее состояние параметра сохраняется отдельно, чтобы модель видела последнюю известную точку ряда.
    for column in config.get("state_columns", []):
        if column in frame.columns:
            additions[f"{column}_state"] = pd.to_numeric(frame[column], errors="coerce")

    # Лаги строятся только с положительным сдвигом назад: так в признаках не появляется будущее значение.
    for lag in config.get("lags", []):
        steps = duration_to_steps(lag, dataset_step)
        suffix = safe_name(lag)
        for column in config.get("lag_columns", []):
            if column in frame.columns:
                additions[f"{column}_lag_{suffix}"] = pd.to_numeric(frame[column], errors="coerce").shift(steps)

    # Разности помогают модели видеть скорость изменения целевого параметра на разных временных масштабах.
    for lag in config.get("diff_lags", []):
        steps = duration_to_steps(lag, dataset_step)
        suffix = safe_name(lag)
        for column in config.get("targets", []):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                additions[f"{column}_diff_{suffix}"] = values - values.shift(steps)
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def add_targets(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Добавляет целевые значения для заданных горизонтов прогноза."""
    dataset_step = config["dataset_step"]
    additions: dict[str, pd.Series] = {}
    for horizon in config.get("forecast_horizons", []):
        steps = duration_to_steps(horizon, dataset_step)
        suffix = safe_name(horizon)
        for target in config.get("targets", []):
            if target in frame.columns:
                # Отрицательный сдвиг делает целевую колонку значением параметра в будущем горизонте прогноза.
                additions[f"{target}_target_{suffix}"] = pd.to_numeric(frame[target], errors="coerce").shift(-steps)
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def load_station_metadata(config: dict[str, Any], base_dir: Path) -> pd.DataFrame | None:
    """Загружает метаданные станций и приводит их к табличному виду."""
    if not config.get("include_station_metadata", False):
        return None
    metadata_path = base_dir / config.get("station_metadata_path", "")
    if not metadata_path.exists():
        return None
    metadata = pd.read_csv(metadata_path)
    keep = [column for column in ["station", "latitude", "longitude_180"] if column in metadata.columns]
    return metadata[keep].drop_duplicates("station") if "station" in keep else None


def build_station_features(path: Path, config: dict[str, Any], metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Формирует feature-набор для одной станции."""
    time_column = config.get("time_column", "time_utc")
    station_column = config.get("station_column", "station")
    frame = pd.read_csv(path)
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)

    frame = add_time_features(frame, time_column)
    frame = add_lag_features(frame, config)
    frame = add_targets(frame, config)

    if metadata is not None and station_column in frame.columns:
        frame = frame.merge(metadata, on=station_column, how="left")
    return frame


def feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Выбирает список признаков, используемых для обучения моделей."""
    excluded_prefixes = tuple(f"{target}_target_" for target in config.get("targets", []))
    excluded_exact = {
        config.get("time_column", "time_utc"),
        "interval",
        config.get("station_column", "station"),
    }
    columns = []
    for column in frame.select_dtypes(include=np.number).columns:
        # Служебные временные поля и target-колонки исключаются, чтобы не допустить утечку будущего.
        if column in excluded_exact or column.endswith("_source_time") or column.endswith("_valid_until"):
            continue
        if column.startswith(excluded_prefixes):
            continue
        columns.append(column)
    return columns


def write_manifest(output_dir: Path, config: dict[str, Any], station_summaries: list[dict[str, Any]]) -> None:
    """Сохраняет manifest с описанием сформированных признаков."""
    manifest = {
        "config": config,
        "stations": station_summaries,
        "station_count": len(station_summaries),
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Запускает основной сценарий файла."""
    args = parse_args()
    base_dir = Path.cwd()
    input_dir = Path(args.input_dir)
    station_dir = input_dir / "stations"
    output_dir = Path(args.output_dir)
    output_station_dir = output_dir / "stations"
    output_station_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    metadata = load_station_metadata(config, base_dir)
    paths = sorted(station_dir.glob("*_time_grid.csv"))
    if args.station:
        paths = [path for path in paths if path.name == f"{args.station}_time_grid.csv"]
    if not paths:
        raise FileNotFoundError(f"No station CSV files found in {station_dir}.")

    station_summaries: list[dict[str, Any]] = []
    for path in paths:
        station = path.name.removesuffix("_time_grid.csv")
        frame = build_station_features(path, config, metadata)
        features = feature_columns(frame, config)
        min_coverage = float(config.get("minimum_feature_coverage", 0.0))
        selected_features = [column for column in features if frame[column].notna().mean() >= min_coverage]
        out_path = output_station_dir / f"{station}_features.csv"
        frame.to_csv(out_path, index=False)
        station_summaries.append(
            {
                "station": station,
                "rows": int(len(frame)),
                "feature_columns": selected_features,
                "feature_count": int(len(selected_features)),
                "output": str(out_path),
            }
        )
        print(f"{station}: rows={len(frame)} features={len(selected_features)} -> {out_path}")

    summary = pd.DataFrame(
        [{"station": item["station"], "rows": item["rows"], "feature_count": item["feature_count"]} for item in station_summaries]
    )
    summary.to_csv(output_dir / "feature_summary.csv", index=False)
    write_manifest(output_dir, config, station_summaries)


if __name__ == "__main__":
    main()
