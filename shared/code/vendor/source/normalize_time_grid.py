from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_MARKERS = {"", "---", "__", "nan", "na", "null", "none"}
GIRO_MEASUREMENTS = [
    "foF2",
    "foF1",
    "foE",
    "foEs",
    "fbEs",
    "foEa",
    "foP",
    "fxI",
    "MUF_D",
    "M_D",
    "hF2",
    "hF",
    "hE",
    "hEs",
    "hEa",
    "hP",
    "hmF2",
    "hmF1",
    "hmE",
    "halfNm",
    "yF2",
    "yF1",
    "yE",
    "B0",
    "B1",
    "D1",
    "TEC",
    "FF",
    "FE",
    "QF",
    "QE",
    "fmin",
    "fminF",
    "fminE",
    "fminEs",
    "foF2p",
]
CORE_GIRO_COLUMNS = ["foF2", "foF1", "foE", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"]
CORE_GIRO_POSITIONS = {name: GIRO_MEASUREMENTS.index(name) for name in CORE_GIRO_COLUMNS}


def main() -> None:
    """Запускает основной сценарий файла."""
    parser = argparse.ArgumentParser(description="Normalize collected GIRO and index data to one time grid.")
    parser.add_argument("--giro-raw-dir", default="data_2024_2025_giro/raw/giro")
    parser.add_argument("--processed-dir", default="cleaned_2024_2025/processed")
    parser.add_argument("--output-dir", default="normalized_2024_2025")
    parser.add_argument("--time-config", default="configs/time_normalization.json")
    parser.add_argument("--station-selection-config", default="configs/station_selection.json")
    parser.add_argument("--freq", default="auto", help="Pandas frequency, for example 5min. Use auto to infer.")
    parser.add_argument("--min-cs", type=float, default=50.0, help="Soft GIRO confidence cutoff.")
    parser.add_argument(
        "--max-interpolate-gap",
        default="30min",
        help="Deprecated compatibility option. GIRO values are not interpolated.",
    )
    parser.add_argument("--top-stations", type=int, default=0, help="Normalize only top N stations by raw data rows. 0 means all.")
    parser.add_argument("--station-set", default="", help="JSON file with a stations list to normalize.")
    parser.add_argument("--split-by-station", action="store_true", help="Also write one analytical CSV per station.")
    parser.add_argument("--quality-only", action="store_true", help="Write quality reports and summary without large station CSV outputs.")
    args = parser.parse_args()
    time_config = load_time_config(Path(args.time_config))
    station_selection = load_time_config(Path(args.station_selection_config)) or time_config.get("station_selection", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_station_stats = scan_giro_station_stats(Path(args.giro_raw_dir))
    include_stations = load_station_set(Path(args.station_set)) if args.station_set else choose_top_stations(raw_station_stats, args.top_stations)
    if include_stations is not None:
        print(f"Using top {len(include_stations)} stations by raw coverage: {', '.join(include_stations)}")

    print("Reading GIRO raw files...")
    giro = read_giro_raw(Path(args.giro_raw_dir), include_stations=include_stations, raw_station_stats=raw_station_stats)
    if giro.empty:
        raise SystemExit("No GIRO measurement rows found.")

    coverage = build_station_coverage(giro)
    excluded = coverage[coverage["data_rows"] == 0]["station"].tolist()
    active_stations = sorted(giro["station"].dropna().unique().tolist())
    freq = choose_target_frequency(args.freq, time_config, giro["time_utc"])
    print(f"Inferred/selected frequency: {freq}")
    print(f"Active GIRO stations: {len(active_stations)}")

    print("Normalizing GIRO to time grid...")
    normalized_giro = normalize_giro(giro, freq, args.min_cs, args.max_interpolate_gap)
    if not args.quality_only:
        normalized_giro.to_csv(output_dir / "giro_time_grid.csv", index=False)

    print("Preparing index features...")
    processed_dir = Path(args.processed_dir)
    indices = prepare_index_features(processed_dir, freq, time_config)

    print("Merging features by time...")
    analytical = normalized_giro.merge(indices, on="time_utc", how="left")
    if not args.quality_only:
        analytical.to_csv(output_dir / "analytical_time_grid.csv", index=False)
    station_files = write_station_datasets(analytical, output_dir) if args.split_by_station and not args.quality_only else []

    coverage.to_csv(output_dir / "station_coverage.csv", index=False)
    quality_files, quality_summary = write_station_quality_reports(
        analytical,
        coverage,
        output_dir,
        freq,
        station_selection,
    )
    write_quality_summary(output_dir / "reports" / "station_quality_summary.csv", quality_summary)
    write_report(
        output_dir / "time_normalization_report.md",
        giro,
        normalized_giro,
        analytical,
        coverage,
        excluded,
        active_stations,
        freq,
        args.min_cs,
        args.max_interpolate_gap,
        station_files,
        quality_files,
        args.quality_only,
    )
    print(f"Wrote normalized outputs to {output_dir}")


def load_time_config(path: Path) -> dict[str, Any]:
    """Загружает настройки нормализации временной сетки."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def choose_target_frequency(freq_arg: str, time_config: dict[str, Any], times: pd.Series) -> str:
    """Выбирает целевую частоту временной сетки."""
    if freq_arg != "auto":
        return freq_arg
    configured = time_config.get("target_frequency")
    if configured:
        return str(configured)
    return infer_frequency(times)


def scan_giro_station_stats(raw_dir: Path) -> dict[str, dict[str, int]]:
    """Сканирует GIRO-данные и собирает статистику по станциям."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "data_lines": 0})
    for path in sorted(raw_dir.rglob("*.txt")):
        station = get_path_value(path, "station")
        stats[station]["files"] += 1
        data_lines = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and not stripped.upper().startswith("ERROR:"):
                    data_lines += 1
        stats[station]["data_lines"] += data_lines
    return dict(stats)


def choose_top_stations(raw_station_stats: dict[str, dict[str, int]], top_stations: int) -> set[str] | None:
    """Выбирает станции с наилучшим покрытием данных."""
    if top_stations <= 0:
        return None
    ranked = sorted(
        ((station, stats["data_lines"]) for station, stats in raw_station_stats.items() if stats["data_lines"] > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return {station for station, _ in ranked[:top_stations]}


def load_station_set(path: Path) -> set[str]:
    """Загружает список станций для обработки."""
    payload = load_time_config(path)
    stations = payload.get("stations", [])
    if not stations:
        raise ValueError(f"Station set has no stations: {path}")
    return {str(station) for station in stations}


def read_giro_raw(
    raw_dir: Path,
    include_stations: set[str] | None = None,
    raw_station_stats: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Читает сырые GIRO-файлы в общий табличный набор."""
    rows: list[dict[str, Any]] = []
    raw_station_stats = raw_station_stats or {}
    for path in sorted(raw_dir.rglob("*.txt")):
        station = get_path_value(path, "station")
        if include_stations is not None and station not in include_stations:
            continue
        year = get_path_value(path, "year")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.upper().startswith("ERROR:"):
                    continue
                parsed = parse_giro_line(stripped)
                if parsed:
                    parsed["station"] = station
                    parsed["year"] = year
                    rows.append(parsed)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"])
    numeric_columns = ["CS", *CORE_GIRO_COLUMNS]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df.attrs["raw_station_stats"] = dict(raw_station_stats)
    return df.sort_values(["station", "time_utc"])


def parse_giro_line(line: str) -> dict[str, Any] | None:
    """Разбирает одну строку GIRO-данных."""
    parts = re.split(r"\s+", line.strip())
    if len(parts) < 3:
        return None
    row: dict[str, Any] = {"time_utc": parts[0], "CS": to_number(parts[1])}
    for measurement, measurement_index in CORE_GIRO_POSITIONS.items():
        token_index = 2 + measurement_index * 2
        row[measurement] = to_number(parts[token_index]) if token_index < len(parts) else None
    return row


def to_number(value: Any) -> float | None:
    """Преобразует значение в число."""
    value = missing_to_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def missing_to_none(value: Any) -> str | None:
    """Заменяет служебные пропуски на None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return None
    return text


def get_path_value(path: Path, key: str) -> str:
    """Извлекает вложенное значение из словаря по пути ключей."""
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return ""


def infer_frequency(times: pd.Series) -> str:
    """Оценивает фактический временной шаг ряда."""
    rounded = pd.to_datetime(times, utc=True).sort_values().drop_duplicates()
    deltas = rounded.diff().dropna().dt.total_seconds()
    deltas = deltas[(deltas > 0) & (deltas <= 3600)]
    if deltas.empty:
        return "5min"
    minute_deltas = (deltas / 60).round().astype(int)
    minute_deltas = minute_deltas[minute_deltas > 0]
    if minute_deltas.empty:
        return "5min"
    mode_minutes = Counter(minute_deltas.tolist()).most_common(1)[0][0]
    return f"{mode_minutes}min"


def normalize_giro(df: pd.DataFrame, freq: str, min_cs: float, max_interpolate_gap: str) -> pd.DataFrame:
    """Приводит GIRO-наблюдения к регулярной временной сетке."""
    # Сначала отбрасываются измерения с плохим CS, чтобы регулярная сетка не строилась на заведомо слабых точках.
    filtered = df[(df["CS"].isna()) | (df["CS"] == -1) | (df["CS"] >= min_cs)].copy()
    numeric_columns = [column for column in ["CS", *CORE_GIRO_COLUMNS] if column in filtered.columns]
    pieces = []
    for station, station_df in filtered.groupby("station", sort=True):
        station_df = station_df.sort_values("time_utc").drop_duplicates("time_utc", keep="last")
        station_df = station_df.set_index("time_utc")
        # Несколько наблюдений внутри одного интервала сворачиваются в среднее значение этого интервала.
        resampled = station_df[numeric_columns].resample(freq).mean()
        original_counts = station_df.resample(freq).size().reindex(resampled.index, fill_value=0)
        exclusive_end = infer_exclusive_period_end(station_df.index)
        if exclusive_end is not None:
            # Правая граница вида 2026-01-01 00:00 не включается в период 2024-2025.
            resampled = resampled[resampled.index < exclusive_end]
            original_counts = original_counts.reindex(resampled.index, fill_value=0)
        resampled["station"] = station
        resampled["giro_rows_in_interval"] = original_counts.astype(int)
        resampled["had_original_giro_row"] = original_counts.gt(0)
        pieces.append(resampled.reset_index())
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    result["interval"] = make_interval(result["time_utc"], freq)
    ordered = ["time_utc", "interval", "station", "had_original_giro_row", "giro_rows_in_interval", *numeric_columns]
    return result[[column for column in ordered if column in result.columns]]


def infer_exclusive_period_end(index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Определяет правую границу временного периода."""
    if index.empty:
        return None
    last = index.max()
    if last.month == 1 and last.day == 1 and last.hour == 0 and last.minute == 0 and last.second == 0:
        return last
    return None


def make_interval(times: pd.Series, freq: str) -> pd.Series:
    """Формирует подпись временного интервала."""
    delta = pd.to_timedelta(freq)
    starts = pd.to_datetime(times, utc=True)
    ends = starts + delta
    return starts.dt.strftime("%H:%M") + "-" + ends.dt.strftime("%H:%M")


def prepare_index_features(processed_dir: Path, freq: str, time_config: dict[str, Any]) -> pd.DataFrame:
    """Подготавливает индексные признаки для совмещения с GIRO."""
    variable_rules = time_config.get("variables", {})
    frames = []
    gfz_path = processed_dir / "geophysical_indices.csv"
    if gfz_path.exists() and gfz_path.stat().st_size > 2:
        gfz = pd.read_csv(gfz_path)
        gfz["time_utc"] = pd.to_datetime(gfz["time_utc"], utc=True, errors="coerce")
        gfz["value"] = pd.to_numeric(gfz["value"], errors="coerce")
        gfz = gfz.dropna(subset=["time_utc", "index"])
        gfz = gfz.pivot_table(index="time_utc", columns="index", values="value", aggfunc="mean")
        gfz = gfz.add_prefix("gfz_").sort_index()
        frames.append(propagate_interval_features(gfz, freq, variable_rules))

    omni_path = processed_dir / "omni_solar_wind.csv"
    if omni_path.exists() and omni_path.stat().st_size > 2:
        omni = pd.read_csv(omni_path)
        omni["time_utc"] = pd.to_datetime(omni["time_utc"], utc=True, errors="coerce")
        for column in ["bz_gsm_nT", "dst_nT", "flow_speed_km_s", "proton_density_n_cc"]:
            if column in omni.columns:
                omni[column] = pd.to_numeric(omni[column], errors="coerce")
        keep = [column for column in ["bz_gsm_nT", "dst_nT", "flow_speed_km_s", "proton_density_n_cc"] if column in omni.columns]
        omni = omni.dropna(subset=["time_utc"]).set_index("time_utc")[keep].sort_index()
        frames.append(propagate_interval_features(omni, freq, variable_rules))

    if not frames:
        return pd.DataFrame({"time_utc": []})

    merged = pd.concat(frames, axis=1).sort_index().reset_index()
    return merged.rename(columns={"index": "time_utc"})


def propagate_interval_features(frame: pd.DataFrame, freq: str, variable_rules: dict[str, Any]) -> pd.DataFrame:
    """Распространяет индексные признаки на интервалы их действия."""
    if frame.empty:
        return frame
    start = frame.index.min().floor(freq)
    end = frame.index.max().ceil(freq)
    grid = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    output = pd.DataFrame(index=grid)
    source_times = pd.Series(frame.index, index=frame.index)

    for column in frame.columns:
        valid_for = pd.to_timedelta(variable_rules.get(column, {}).get("valid_for", "0s"))
        provenance_name = provenance_column_name(column)
        # Индексы вроде Kp/Ap действуют интервалами, поэтому значение переносится вперед только до valid_until.
        values = frame[column].resample(freq).ffill().reindex(grid).ffill()
        propagated_source = source_times.resample(freq).ffill().reindex(grid).ffill()
        valid_until = propagated_source + valid_for
        is_valid = propagated_source.notna() & (valid_for > pd.Timedelta(0)) & (output.index < valid_until)

        output[column] = values.where(is_valid)
        output[f"{provenance_name}_source_time"] = propagated_source.where(is_valid)
        output[f"{provenance_name}_valid_until"] = valid_until.where(is_valid)
    return output


def provenance_column_name(column: str) -> str:
    """Формирует имя столбца с источником или временем происхождения признака."""
    aliases = {
        "gfz_Kp": "Kp",
        "gfz_ap": "ap",
        "gfz_Ap": "Ap",
        "gfz_Fobs": "Fobs",
        "gfz_Fadj": "Fadj",
        "bz_gsm_nT": "Bz",
        "dst_nT": "Dst",
        "flow_speed_km_s": "flow_speed",
        "proton_density_n_cc": "proton_density",
    }
    return aliases.get(column, column)


def write_station_datasets(analytical: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Сохраняет нормализованные наборы данных по станциям."""
    station_dir = output_dir / "stations"
    station_dir.mkdir(parents=True, exist_ok=True)
    station_files = []
    for station, station_df in analytical.groupby("station", sort=True):
        path = station_dir / f"{station}_time_grid.csv"
        station_df = station_df.sort_values("time_utc").reset_index(drop=True)
        station_df.to_csv(path, index=False)
        station_files.append(path)
    return station_files


def write_station_quality_reports(
    analytical: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
    freq: str,
    selection_rules: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Создает отчеты качества по каждой станции."""
    report_dir = output_dir / "reports" / "station_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    files = []
    summary_rows: list[dict[str, Any]] = []
    coverage_by_station = coverage.set_index("station").to_dict("index") if not coverage.empty else {}
    target = selection_rules.get("target", "foF2")
    min_original = float(selection_rules.get("minimum_original_coverage_percent", 50.0))
    max_missing = float(selection_rules.get("maximum_missing_percent", 40.0))
    hard_exclude_only_if_no_target = bool(selection_rules.get("hard_exclude_only_if_no_target_data", False))
    min_years = int(selection_rules.get("minimum_years_present", 1))
    min_seasons = int(selection_rules.get("minimum_seasons_present", 1))
    max_gap_days = float(selection_rules.get("maximum_longest_gap_days", 3650.0))
    require_test_period = bool(selection_rules.get("require_test_period_data", False))
    test_period = selection_rules.get("test_period", {})

    for station, station_df in analytical.groupby("station", sort=True):
        station_df = station_df.sort_values("time_utc").reset_index(drop=True)
        station_df["time_utc"] = pd.to_datetime(station_df["time_utc"], utc=True)
        total_rows = len(station_df)
        target_present = station_df[target].notna() if target in station_df.columns else pd.Series(False, index=station_df.index)
        original_mask = station_df["had_original_giro_row"].astype(bool) & target_present
        missing_mask = ~target_present
        original_percent = percent(original_mask.sum(), total_rows)
        missing_percent = percent(missing_mask.sum(), total_rows)
        longest_gap_hours = longest_missing_gap_hours(missing_mask, freq)
        coverage_by_year = grouped_coverage(station_df, target_present, station_df["time_utc"].dt.year.astype(str))
        coverage_by_season = grouped_coverage(station_df, target_present, station_df["time_utc"].dt.month.map(season_name))
        years_present = sum(1 for item in coverage_by_year.values() if item["original_rows"] > 0)
        seasons_present = sum(1 for item in coverage_by_season.values() if item["original_rows"] > 0)
        test_coverage = period_coverage(station_df, target_present, test_period)
        cs_distribution = cs_summary(station_df)
        warnings = []
        if original_percent < min_original:
            warnings.append(f"{target}_original_coverage_below_{min_original:g}_percent")
        if missing_percent > max_missing:
            warnings.append(f"{target}_missing_above_{max_missing:g}_percent")
        if years_present < min_years:
            warnings.append(f"years_present_below_{min_years}")
        if seasons_present < min_seasons:
            warnings.append(f"seasons_present_below_{min_seasons}")
        if longest_gap_hours / 24 > max_gap_days:
            warnings.append(f"longest_gap_above_{max_gap_days:g}_days")
        if require_test_period and test_coverage["original_rows"] == 0:
            warnings.append("missing_test_period_data")
        hard_exclusion_reasons = []
        if int(original_mask.sum()) == 0:
            hard_exclusion_reasons.append(f"no_original_{target}_data")
        quality_class = classify_station_quality(original_percent, missing_percent, selection_rules)
        coverage_row = coverage_by_station.get(station, {})
        report = {
            "station_id": station,
            "period": {
                "start": stringify_timestamp(station_df["time_utc"].min()),
                "end": stringify_timestamp(station_df["time_utc"].max()),
            },
            "raw_files": int(coverage_row.get("raw_files", 0) or 0),
            "source_measurement_rows": int(coverage_row.get("data_rows", 0) or 0),
            "normalized_rows": total_rows,
            target: {
                "original_percent": original_percent,
                "missing_percent": missing_percent,
                "longest_gap_hours": longest_gap_hours,
                "coverage_by_year": coverage_by_year,
                "coverage_by_season": coverage_by_season,
                "test_period_coverage": test_coverage,
            },
            "cs_distribution": cs_distribution,
            "quality_class": quality_class,
            "recommended_for_exploration": not hard_exclusion_reasons and quality_class in {"good", "usable", "weak"},
            "recommended_for_strict_dataset": not warnings and not hard_exclusion_reasons,
            "selection_rules": selection_rules,
            "accepted": not hard_exclusion_reasons if hard_exclude_only_if_no_target else not warnings and not hard_exclusion_reasons,
            "warnings": warnings,
            "exclusion_reasons": hard_exclusion_reasons,
        }
        path = report_dir / f"{station}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        files.append(path)
        summary_rows.append(
            {
                "station": station,
                "quality_class": quality_class,
                "accepted": report["accepted"],
                "recommended_for_exploration": report["recommended_for_exploration"],
                "recommended_for_strict_dataset": report["recommended_for_strict_dataset"],
                "normalized_rows": total_rows,
                "source_measurement_rows": report["source_measurement_rows"],
                f"{target}_original_percent": original_percent,
                f"{target}_missing_percent": missing_percent,
                f"{target}_longest_gap_hours": longest_gap_hours,
                "years_with_data": years_present,
                "seasons_with_data": seasons_present,
                "test_original_percent": test_coverage["original_percent"],
                "warnings": ";".join(warnings),
                "exclusion_reasons": ";".join(hard_exclusion_reasons),
            }
        )
    return files, summary_rows


def write_quality_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Сохраняет сводную таблицу качества станций."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["quality_class", "station"]).to_csv(path, index=False)


def percent(count: int, total: int) -> float:
    """Переводит отношение в проценты."""
    return round(count / total * 100, 2) if total else 0.0


def classify_station_quality(original_percent: float, missing_percent: float, selection_rules: dict[str, Any]) -> str:
    """Присваивает станции класс качества по покрытию данных."""
    classes = selection_rules.get("quality_classes", {})
    for name in ("good", "usable", "weak"):
        rule = classes.get(name, {})
        min_original = float(rule.get("minimum_original_coverage_percent", -1.0))
        max_missing = float(rule.get("maximum_missing_percent", 101.0))
        if original_percent >= min_original and missing_percent <= max_missing:
            return name
    return "exclude"


def longest_missing_gap_hours(mask: pd.Series, freq: str) -> float:
    """Вычисляет максимальную длину пропуска во временном ряду."""
    if mask.empty:
        return 0.0
    max_run = 0
    current_run = 0
    for missing in mask.tolist():
        if missing:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return round(max_run * pd.to_timedelta(freq).total_seconds() / 3600, 2)


def grouped_coverage(df: pd.DataFrame, present_mask: pd.Series, groups: pd.Series) -> dict[str, dict[str, Any]]:
    """Считает покрытие данных по группам."""
    result: dict[str, dict[str, Any]] = {}
    original_mask = df["had_original_giro_row"].astype(bool) & present_mask
    valid_groups = [group for group in groups.dropna().unique().tolist() if group != "2026"]
    for group in sorted(valid_groups):
        group_mask = groups == group
        total = int(group_mask.sum())
        original_rows = int((group_mask & original_mask).sum())
        missing_rows = int((group_mask & ~present_mask).sum())
        result[str(group)] = {
            "rows": total,
            "original_rows": original_rows,
            "missing_rows": missing_rows,
            "original_percent": percent(original_rows, total),
            "missing_percent": percent(missing_rows, total),
        }
    return result


def period_coverage(df: pd.DataFrame, present_mask: pd.Series, period: dict[str, Any]) -> dict[str, Any]:
    """Считает покрытие данных за заданный период."""
    start = pd.to_datetime(period.get("start"), utc=True, errors="coerce")
    end = pd.to_datetime(period.get("end"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return {"rows": 0, "original_rows": 0, "missing_rows": 0, "original_percent": 0.0, "missing_percent": 0.0}
    period_mask = (df["time_utc"] >= start) & (df["time_utc"] < end)
    original_mask = df["had_original_giro_row"].astype(bool) & present_mask
    total = int(period_mask.sum())
    original_rows = int((period_mask & original_mask).sum())
    missing_rows = int((period_mask & ~present_mask).sum())
    return {
        "start": stringify_timestamp(start),
        "end": stringify_timestamp(end),
        "rows": total,
        "original_rows": original_rows,
        "missing_rows": missing_rows,
        "original_percent": percent(original_rows, total),
        "missing_percent": percent(missing_rows, total),
    }


def season_name(month: int) -> str:
    """Возвращает название сезона по номеру месяца."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def cs_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Собирает статистику качества ионосферных измерений по CS."""
    if "CS" not in df.columns:
        return {}
    cs = pd.to_numeric(df["CS"], errors="coerce").dropna()
    if cs.empty:
        return {"known_rows": 0}
    return {
        "known_rows": int(len(cs)),
        "unknown_minus_one_rows": int((cs == -1).sum()),
        "lt_50_rows": int(((cs >= 0) & (cs < 50)).sum()),
        "gte_50_rows": int((cs >= 50).sum()),
        "median": round(float(cs.median()), 2),
        "min": round(float(cs.min()), 2),
        "max": round(float(cs.max()), 2),
    }


def stringify_timestamp(value: Any) -> str:
    """Преобразует timestamp в строку для отчетов."""
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def build_station_coverage(giro: pd.DataFrame) -> pd.DataFrame:
    """Формирует сводку покрытия данных по станциям."""
    raw_stats = giro.attrs.get("raw_station_stats", {})
    if giro.empty:
        rows = [{"station": station, "data_rows": stats["data_lines"], "raw_files": stats["files"], "start_utc": pd.NaT, "end_utc": pd.NaT} for station, stats in raw_stats.items()]
        return pd.DataFrame(rows, columns=["station", "data_rows", "raw_files", "start_utc", "end_utc"])
    coverage = (
        giro.groupby("station")
        .agg(data_rows=("time_utc", "size"), start_utc=("time_utc", "min"), end_utc=("time_utc", "max"))
        .reset_index()
    )
    coverage["raw_files"] = coverage["station"].map(lambda station: raw_stats.get(station, {}).get("files", 0))
    existing = set(coverage["station"])
    zero_rows = [
        {"station": station, "data_rows": stats["data_lines"], "raw_files": stats["files"], "start_utc": pd.NaT, "end_utc": pd.NaT}
        for station, stats in raw_stats.items()
        if station not in existing
    ]
    if zero_rows:
        coverage = pd.concat([coverage, pd.DataFrame(zero_rows)], ignore_index=True)
    coverage = coverage.sort_values(["data_rows", "station"], ascending=[False, True])
    return coverage


def write_report(
    path: Path,
    giro: pd.DataFrame,
    normalized: pd.DataFrame,
    analytical: pd.DataFrame,
    coverage: pd.DataFrame,
    excluded: list[str],
    active_stations: list[str],
    freq: str,
    min_cs: float,
    max_interpolate_gap: str,
    station_files: list[Path],
    quality_files: list[Path],
    quality_only: bool = False,
) -> None:
    """Формирует текстовый отчет по результатам обработки."""
    source_missing = missing_summary(giro, ["foF2", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"])
    normalized_missing = missing_summary(normalized, ["foF2", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"])
    lines = [
        "# Time normalization report",
        "",
        f"- Selected time step: `{freq}`.",
        f"- GIRO confidence filter: `CS >= {min_cs}`, `CS = -1`, or missing CS.",
        "- GIRO normalization: measurements are aggregated inside each time interval; missing GIRO intervals stay empty.",
        "- Index normalization: GFZ/OMNI values are carried forward over their valid time intervals.",
        f"- Active stations used for training grid: {len(active_stations)}.",
        f"- Stations excluded because they have no measurement rows in parsed GIRO: {len(excluded)}.",
        f"- Quality-only mode: {quality_only}.",
        "",
        "## Output files",
        "- `giro_time_grid.csv`: normalized GIRO rows by station and time." if not quality_only else "- `giro_time_grid.csv`: skipped in quality-only mode.",
        "- `analytical_time_grid.csv`: GIRO grid with geophysical and OMNI features joined by time." if not quality_only else "- `analytical_time_grid.csv`: skipped in quality-only mode.",
        f"- `stations/*.csv`: {len(station_files)} separate station datasets.",
        f"- `reports/station_quality/*.json`: {len(quality_files)} station quality reports.",
        "- `reports/station_quality_summary.csv`: compact station quality summary.",
        "- `station_coverage.csv`: station coverage summary.",
        "",
        "## Shape",
        f"- Parsed GIRO source rows: {len(giro)}.",
        f"- Normalized GIRO rows: {len(normalized)}.",
        f"- Analytical rows: {len(analytical)}.",
        "",
        "## Missing values before normalization",
        format_missing(source_missing),
        "",
        "## Missing values after normalization",
        format_missing(normalized_missing),
        "",
        "## Excluded stations",
        ", ".join(excluded) if excluded else "None in parsed GIRO table.",
        "",
        "## Top stations by rows",
    ]
    for _, row in coverage.head(20).iterrows():
        lines.append(f"- {row['station']}: {int(row['data_rows'])} rows, {row['start_utc']} - {row['end_utc']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def missing_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Формирует краткое описание пропусков по столбцам."""
    if df.empty:
        return {}
    result = {}
    for column in columns:
        if column in df.columns:
            result[column] = round(df[column].isna().mean() * 100, 2)
    return result


def format_missing(values: dict[str, float]) -> str:
    """Форматирует информацию о пропусках для отчета."""
    if not values:
        return "No values."
    return "\n".join(f"- {column}: {percent:.2f}%" for column, percent in values.items())


if __name__ == "__main__":
    main()
