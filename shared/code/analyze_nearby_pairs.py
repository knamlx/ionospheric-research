"""Descriptive nearby-station preflight. No model fitting or forecast evaluation.

python shared/code/analyze_nearby_pairs.py

Reads hash-checked GIRO responses listed in data/manifest.json. The completed
three-station experiment is not imported or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAIRS = [("PQ052", "SO148"), ("IC437", "JJ433")]
START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
GRID = pd.date_range(START, END, freq="15min", inclusive="left", name="time_utc")
SEED = 20261102


def finite_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def load_observations(root, manifest):
    frames, audits = {}, []
    for station in dict.fromkeys(s for pair in PAIRS for s in pair):
        rows = []
        for item in manifest["sources"]:
            if item["station"] != station:
                continue
            path = root / item["file"]
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"], path
            assert ("URSI-Code " + station).encode() in data, path
            assert int(item["year"]) in (2024, 2025)
            for line in data.decode("latin-1").splitlines():
                parts = line.split()
                if len(parts) != 4 or not parts[0].startswith(("2024-", "2025-", "2026-")):
                    continue
                rows.append((parts[0], finite_float(parts[1]), finite_float(parts[2])))
        frame = pd.DataFrame(rows, columns=["time", "cs", "foF2"])
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        # Boundary measurements can occur in both half-year responses.
        frame = frame[(frame.time >= START) & (frame.time < END)]
        duplicated = int(frame.time.duplicated().sum())
        conflicting = int(frame.groupby("time").foF2.nunique().gt(1).sum())
        assert conflicting == 0, (station, "Conflicting duplicate timestamps", conflicting)
        frame = frame.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
        frame = frame.set_index("time")
        physical = frame.foF2.between(.5, 25)
        audits.append({"station": station, "unique_observation_rows": len(frame),
                       "boundary_duplicates_removed": duplicated,
                       "conflicting_duplicate_timestamps": conflicting,
                       "physical_foF2_rows": int(physical.sum()),
                       "unknown_cs_rows": int((frame.cs.isna() | frame.cs.eq(-1)).sum()),
                       "low_cs_physical_rows": int((physical & frame.cs.lt(50)).sum()),
                       "invalid_foF2_rows": int((~physical).sum())})
        frames[station] = frame
    return frames, audits


def panel_for_cs(frames, minimum_cs):
    result = {}
    for station, frame in frames.items():
        selected = frame.foF2.between(.5, 25) & frame.cs.ge(minimum_cs)
        result[station] = frame.loc[selected, "foF2"].resample("15min").mean().reindex(GRID)
    return pd.DataFrame(result, index=GRID)


def past_baseline(panel):
    """For each UT slot, use only observations from previous calendar days."""
    result = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for station in panel:
        values = panel[station].to_numpy().reshape(-1, 96)
        history = pd.DataFrame(values).shift(1).rolling(27, min_periods=7).median()
        result[station] = history.to_numpy().reshape(-1)
    return result


def corr(x, y, rank=False):
    sample = pd.concat([x, y], axis=1).dropna()
    n = len(sample)
    if n < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return np.nan, n
    if rank:
        sample = sample.rank(method="average")
    return float(np.corrcoef(sample.to_numpy().T)[0, 1]), n


def calendar_block_ci(x, y, repetitions, block_days=7):
    """Circular paired calendar-day bootstrap using daily sufficient statistics."""
    xv, yv = x.to_numpy(), y.to_numpy()
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = np.where(valid, xv, 0.), np.where(valid, yv, 0.)
    stats = np.column_stack([valid.astype(float), xv, yv, xv*xv, yv*yv, xv*yv])
    daily = stats.reshape(-1, 96, 6).sum(axis=1)
    n_days = len(daily)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(repetitions):
        starts = rng.integers(0, n_days, size=math.ceil(n_days / block_days))
        days = ((starts[:, None] + np.arange(block_days)) % n_days).ravel()[:n_days]
        n, sx, sy, sxx, syy, sxy = daily[days].sum(axis=0)
        vx, vy = sxx-sx*sx/n, syy-sy*sy/n
        if n > 2 and vx > 0 and vy > 0:
            boot.append((sxy-sx*sy/n)/math.sqrt(vx*vy))
    low, high = np.quantile(boot, [.025, .975])
    return float(low), float(high)


def longest_missing(series):
    mask = series.isna().to_numpy() if isinstance(series, pd.Series) else ~np.asarray(series)
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return {"bins": 0, "hours": 0., "start": None, "end_exclusive": None}
    i = int(np.argmax(stops-starts))
    return {"bins": int(stops[i]-starts[i]), "hours": float((stops[i]-starts[i])/4),
            "start": str(GRID[starts[i]]),
            "end_exclusive": str(END if stops[i] == len(GRID) else GRID[stops[i]])}


def partial_corr(sample, number_controls):
    """Conditional covariance (Schur complement), not a fitted forecast model."""
    complete = sample.dropna().to_numpy(dtype=float)
    if len(complete) < number_controls + 4:
        return np.nan, len(complete)
    covariance = np.cov(complete, rowvar=False)
    conditional = (covariance[:2, :2] - covariance[:2, 2:]
                   @ np.linalg.pinv(covariance[2:, 2:]) @ covariance[2:, :2])
    scale = math.sqrt(max(0., conditional[0, 0] * conditional[1, 1]))
    return float(conditional[0, 1]/scale) if scale else np.nan, len(complete)


def audit_computations(panel, baseline, frames, manifest):
    assert len(panel) == 70176 and panel.index.is_unique and panel.index.is_monotonic_increasing
    assert panel.index.min() >= START and panel.index.max() < END
    for station in panel:
        assert int(panel[station].notna().sum()) == manifest["expected_valid_bins"][station]
    for a, b in PAIRS:
        key = a + "_" + b
        reverse_key = b + "_" + a
        expected = manifest["expected_pair_bins"].get(key, manifest["expected_pair_bins"].get(reverse_key))
        assert int(panel[[a, b]].notna().all(axis=1).sum()) == expected
    # A future observation cannot alter a baseline at or before its own date.
    changed = panel.copy()
    cut = pd.Timestamp("2025-05-15", tz="UTC")
    changed.loc[changed.index >= cut] = 12345.
    changed_baseline = past_baseline(changed)
    before_next_day = baseline.index < cut + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(baseline.loc[before_next_day], changed_baseline.loc[before_next_day])
    # Delta is undefined on either side of an absent interval.
    probe = panel.iloc[:12, :1].copy()
    probe.iloc[5, 0] = np.nan
    assert probe.diff().iloc[5:7, 0].isna().all()
    # Check aggregation against independent floor/groupby, not resample.
    for station, frame in frames.items():
        accepted = frame.loc[frame.foF2.between(.5, 25) & frame.cs.ge(50), "foF2"]
        independent = accepted.groupby(accepted.index.floor("15min")).mean().reindex(GRID)
        np.testing.assert_allclose(panel[station], independent, equal_nan=True)
    return {"checks": ["source_hashes_and_station_headers", "full_2024_2025_grid_including_leap_day",
                       "no_2026_measurements", "independent_coverage_counts",
                       "no_conflicting_duplicates", "independent_bin_mean_aggregation",
                       "baseline_has_no_future_dependency", "differences_do_not_bridge_missing_bins"],
            "status": "passed", "forecast_models_run": False}


def save_figure(out, pair, transforms, summary, monthly, lags):
    a, b = pair
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, transform, title in [(axes[0, 0], "raw", "Observed foF2"),
                                  (axes[0, 1], "anomaly", "Departures from past 27-day UT median")]:
        sample = transforms[transform][[a, b]].dropna()
        ax.hexbin(sample[a], sample[b], gridsize=65, mincnt=1, bins="log", cmap="viridis")
        row = next(r for r in summary if r["transform"] == transform and r["period"] == "2024-2025")
        ax.set(title=f"{title}\nr = {row['pearson']:.3f}, n = {row['n']:,}",
               xlabel=a+" (MHz)", ylabel=b+" (MHz)")
    colors = {"raw": "#265aa8", "anomaly": "#df8b18", "delta15": "#30856d"}
    for transform, color in colors.items():
        lag = [r for r in lags if r["target"] == a and r["transform"] == transform]
        axes[1, 0].plot([r["lag_minutes"] for r in lag], [r["pearson"] for r in lag],
                         marker=".", label=transform, color=color)
        month = [r for r in monthly if r["transform"] == transform]
        axes[1, 1].plot(pd.to_datetime([r["period"] for r in month]),
                         [r["pearson"] for r in month], marker=".", label=transform, color=color)
    axes[1, 0].axvspan(-120, 0, alpha=.06, color="red")
    axes[1, 0].axvline(0, color="gray", lw=.8)
    axes[1, 0].set(xlabel=f"Lag (min): corr({a}(t), {b}(t - lag))", ylabel="Pearson r",
                   title="Lag dependence; shaded region uses future donor")
    axes[1, 1].axhline(0, color="gray", lw=.8)
    axes[1, 1].set(title="Monthly correlation stability", ylabel="Pearson r")
    axes[1, 1].tick_params(axis="x", rotation=30)
    for ax in axes.ravel():
        ax.grid(alpha=.15)
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].legend(fontsize=9)
    fig.suptitle(f"{a} - {b} | 2024-2025 | CS >= 50 | no gap filling", fontsize=14)
    fig.savefig(out / f"{a}_{b}_diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO/"data")
    parser.add_argument("--output", type=Path, default=REPO/"shared/diagnostics/nearby_pairs_2024_2025")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    assert args.bootstrap >= 200
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data/"manifest.json").read_text(encoding="utf-8"))
    frames, audits = load_observations(args.data, manifest)
    panel = panel_for_cs(frames, 50)
    baseline = past_baseline(panel)
    transforms = {"raw": panel, "anomaly": panel-baseline, "delta15": panel.diff()}
    checks = audit_computations(panel, baseline, frames, manifest)
    panel.to_csv(args.data/"observations_15min.csv", float_format="%.9g", lineterminator="\n")
    (out/"sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    pd.DataFrame(audits).to_csv(out/"source_quality_audit.csv", index=False, lineterminator="\n")
    coverage, summary, monthly, lags, partial, sensitivity = [], [], [], [], [], []
    for station in panel:
        row = {"station": station, "valid_bins": int(panel[station].notna().sum()),
               "coverage_pct": float(100*panel[station].notna().mean()),
               "longest_gap": longest_missing(panel[station])}
        for year in (2024, 2025):
            row[str(year)+"_coverage_pct"] = float(100*panel.loc[str(year), station].notna().mean())
        coverage.append(row)
    month_keys = panel.index.strftime("%Y-%m")
    for a, b in PAIRS:
        pair_name = a+"_"+b
        pair_summary, pair_monthly, pair_lags = [], [], []
        for transform, values in transforms.items():
            for period in ("2024-2025", "2024", "2025"):
                subset = values if period == "2024-2025" else values.loc[period]
                r, n = corr(subset[a], subset[b])
                rho, _ = corr(subset[a], subset[b], rank=True)
                row = {"pair": pair_name, "transform": transform, "period": period,
                       "pearson": r, "spearman": rho, "n": n}
                if period == "2024-2025":
                    lo, hi = calendar_block_ci(values[a], values[b], args.bootstrap)
                    row.update(pearson_ci95_low=lo, pearson_ci95_high=hi)
                summary.append(row)
                pair_summary.append(row)
            for month in np.unique(month_keys):
                subset = values.loc[month_keys == month]
                r, n = corr(subset[a], subset[b])
                row = {"pair": pair_name, "transform": transform, "period": month,
                       "pearson": r, "n": n, "joint_coverage_pct": 100*n/len(subset)}
                monthly.append(row)
                pair_monthly.append(row)
        for target, donor in ((a, b), (b, a)):
            for transform, values in transforms.items():
                shifted = {minutes: values[donor].shift(minutes//15)
                           for minutes in range(-120, 121, 15)}
                # Keep exactly the same timestamps across the entire lag profile.
                # This mask is descriptive, not eligibility for a future forecast.
                common = values[target].notna() & pd.DataFrame(shifted).notna().all(axis=1)
                for minutes, donor_values in shifted.items():
                    pairwise_r, pairwise_n = corr(values[target], donor_values)
                    r, n = corr(values[target].where(common), donor_values.where(common))
                    row = {"pair": pair_name, "target": target, "donor": donor,
                           "transform": transform, "lag_minutes": minutes, "pearson": r, "n": n,
                           "pairwise_pearson": pairwise_r, "pairwise_n": pairwise_n,
                           "sample": "Identical timestamps at all lags -120..120 minutes",
                           "uses_future_donor": minutes < 0}
                    lags.append(row)
                    pair_lags.append(row)
            # A next-bin target cannot be scored outside 2025; shift respects the fixed grid.
            sample = pd.DataFrame({"next_target": panel[target].shift(-1),
                                   "donor_now": panel[donor], "target_now": panel[target],
                                   "target_previous": panel[target].shift(1),
                                   "target_past_baseline": baseline[target],
                                   "donor_past_baseline": baseline[donor]})
            for period in ("2024-2025", "2024", "2025"):
                subset = sample if period == "2024-2025" else sample.loc[period].copy()
                if period in ("2024", "2025"):
                    subset.iloc[-1, subset.columns.get_loc("next_target")] = np.nan
                for controls in (1, 4):
                    partial_r, n = partial_corr(subset.iloc[:, :2+controls], controls)
                    partial.append({"pair": pair_name, "target": target, "donor": donor,
                                    "period": period, "horizon_minutes": 15, "controls": controls,
                                    "partial_pearson": partial_r, "n": n,
                                    "definition": "Conditional covariance, not forecast fitting"})
        save_figure(out, (a, b), transforms, pair_summary, pair_monthly, pair_lags)
        print(json.dumps({"pair_completed": pair_name,
                          "correlations": [r for r in pair_summary if r["period"] == "2024-2025"]}), flush=True)
    for cs in (50, 70, 90):
        stricter = panel_for_cs(frames, cs)
        variants = {"raw": stricter, "anomaly": stricter-past_baseline(stricter), "delta15": stricter.diff()}
        for a, b in PAIRS:
            for transform, values in variants.items():
                r, n = corr(values[a], values[b])
                sensitivity.append({"pair": a+"_"+b, "minimum_cs": cs,
                                    "transform": transform, "pearson": r, "n": n})
    for filename, rows in (("correlations.csv", summary), ("monthly_correlations.csv", monthly),
                           ("lag_correlations.csv", lags), ("partial_correlations.csv", partial),
                           ("cs_sensitivity.csv", sensitivity)):
        pd.DataFrame(rows).to_csv(out/filename, index=False, float_format="%.9g", lineterminator="\n")
    pair_coverage = []
    for a, b in PAIRS:
        common = panel[[a, b]].notna().all(axis=1)
        levels = (panel[a]-panel[b]).dropna()
        pair_coverage.append({"pair": a+"_"+b, "common_bins": int(common.sum()),
                              "common_coverage_pct": float(100*common.mean()),
                              "longest_joint_gap": longest_missing(common.to_numpy()),
                              "mean_a_minus_b_mhz": float(levels.mean()),
                              "median_absolute_level_difference_mhz": float(levels.abs().median())})
    result = {"period_start": str(START), "period_end_exclusive": str(END),
              "observed_bins": len(GRID), "station_coverage": coverage,
              "pair_coverage": pair_coverage, "correlations": summary,
              "partial_correlations": partial, "quality_sensitivity": sensitivity,
              "bootstrap": {"repetitions": args.bootstrap, "block_calendar_days": 7, "seed": SEED},
              "validation": checks, "forecast_models_run": False}
    (out/"diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_report(out, result, monthly, lags)
    print(json.dumps({"output": str(out), "validation": checks, "forecast_models_run": False}), flush=True)


def write_report(out, result, monthly, lags):
    lines = ["# Предварительная диагностика близких станций, 2024–2025", "",
             "Статус: выполнена описательная диагностика. Обучение моделей, расчёт прогнозов и ошибок MAE не запускались.", "",
             "Период: 01.01.2024–31.12.2025, UTC, включая 29 февраля. Все измерения 2026 года исключены. "
             "Исходные ответы GIRO проверены по SHA-256 и коду станции. Принимаются конечные foF2 0,5–25 МГц "
             "и CS ≥ 50, включая ручную обработку CS=999. CS=-1 и неизвестное качество исключены. "
             "После фильтрации берётся среднее внутри каждого 15-минутного интервала; пропуски не заполняются.", "",
             "## Покрытие", "", "| Станция | 2024, % | 2025, % | Всего, % | Наибольший пропуск, ч |",
             "|---|---:|---:|---:|---:|"]
    for row in result["station_coverage"]:
        lines.append(f"| {row['station']} | {row['2024_coverage_pct']:.2f} | {row['2025_coverage_pct']:.2f} | "
                     f"{row['coverage_pct']:.2f} | {row['longest_gap']['hours']:.2f} |")
    lines += ["", "| Пара | Расстояние, км | Общие интервалы | Совместное покрытие, % | Наибольший разрыв совместного ряда, ч |",
              "|---|---:|---:|---:|---:|"]
    distances = {"PQ052_SO148": 308.0, "IC437_JJ433": 425.6}
    for row in result["pair_coverage"]:
        lines.append(f"| {row['pair']} | {distances[row['pair']]:.1f} | {row['common_bins']} | "
                     f"{row['common_coverage_pct']:.2f} | {row['longest_joint_gap']['hours']:.2f} |")
    lines += ["", "Разрыв совместного ряда означает отсутствие наблюдений хотя бы одной станции. "
              "Корреляции относятся к одновременно доступным наблюдениям; пропуски не обязательно случайны.",
              "", "## Корреляции", "", "| Пара | Представление | Pearson r | 95% ДИ r | Spearman ρ | Общие интервалы |",
              "|---|---|---:|---|---:|---:|"]
    names = {"raw": "Исходный foF2", "anomaly": "Отклонения от прошлой медианы", "delta15": "Изменения за 15 минут"}
    for row in result["correlations"]:
        if row["period"] == "2024-2025":
            lines.append(f"| {row['pair']} | {names[row['transform']]} | {row['pearson']:.4f} | "
                         f"[{row['pearson_ci95_low']:.4f}; {row['pearson_ci95_high']:.4f}] | "
                         f"{row['spearman']:.4f} | {row['n']} |")
    lines += ["", "Исходная корреляция может включать общий суточный и сезонный ход. Для отклонений каждая станция "
              "сравнивается с медианой того же UTC-интервала за предыдущие 27 календарных дней, минимум 7 доступных "
              "значений. Текущий день и будущее в базовую линию не входят. Это локальная оценка фона, а не физическая "
              "модель ионосферы. Разности вычисляются только между соседними исходными интервалами, без переходов через пропуски.", "",
              "Доверительные интервалы Pearson получены парным круговым bootstrap календарных 7-дневных блоков "
              f"({result['bootstrap']['repetitions']} повторов). Они отражают неопределённость при выбранной длине блока, "
              "но не доказывают причинность и не корректируют все множественные сравнения. Наивные p-value для независимых точек не используются.", "",
              "## Изменение связи между годами", "", "| Пара | Представление | r за 2024 | r за 2025 |",
              "|---|---|---:|---:|"]
    for pair in ("PQ052_SO148", "IC437_JJ433"):
        for transform in names:
            rows = {r["period"]: r for r in result["correlations"] if r["pair"] == pair and r["transform"] == transform}
            lines.append(f"| {pair} | {names[transform]} | {rows['2024']['pearson']:.4f} | {rows['2025']['pearson']:.4f} |")
    lines += ["", "## Помесячная устойчивость", "", "| Пара | Представление | Минимальный r | Максимальный r | Месяцев с r < 0 |",
              "|---|---|---:|---:|---:|"]
    for pair in ("PQ052_SO148", "IC437_JJ433"):
        for transform in names:
            vals = [r["pearson"] for r in monthly if r["pair"] == pair and r["transform"] == transform]
            lines.append(f"| {pair} | {names[transform]} | {min(vals):.4f} | {max(vals):.4f} | {sum(v < 0 for v in vals)} |")
    lines += ["", "## Временные сдвиги", "",
              "Определение: corr(target(t), donor(t − lag)). Положительный сдвиг использует прошлые наблюдения донора; "
              "отрицательный — будущие и не допускается в прогнозных признаках. Внутри каждого профиля сравниваются "
              "одинаковые моменты времени, доступные при всех сдвигах −120…120 минут; поэтому r при нуле может "
              "отличаться от общей таблицы корреляций. Эта диагностическая маска не применяется к будущей оценке "
              "прогноза. Парные оценки на отдельных выборках также сохранены в lag_correlations.csv. Поиск максимума на всём периоде "
              "служит только диагностикой: сдвиг нельзя выбирать по этой таблице и затем объявлять 2025 год независимой проверкой.", "",
              "| Целевая | Донор | Представление | Лучший lag, мин (0…120) | r при нуле | r при лучшем lag |",
              "|---|---|---|---:|---:|---:|"]
    for target, donor in [(a, b) for pair in PAIRS for a, b in (pair, pair[::-1])]:
        for transform in names:
            eligible = [r for r in lags if r["target"] == target and r["transform"] == transform and r["lag_minutes"] >= 0]
            best = max(eligible, key=lambda r: r["pearson"])
            zero = next(r for r in eligible if r["lag_minutes"] == 0)
            lines.append(f"| {target} | {donor} | {names[transform]} | {best['lag_minutes']} | {zero['pearson']:.4f} | {best['pearson']:.4f} |")
    lines += ["", "## Связь со следующим интервалом при известной локальной истории", "",
              "Частная корреляция target(t+15 min) и donor(t) вычислена через условную ковариацию. "
              "В варианте с одним контролем фиксируется target(t); с четырьмя — target(t), target(t−15 min) "
              "и прошлые фоновые медианы обеих станций. Это статистическая диагностика, не обучение прогнозной модели "
              "и не оценка выигрыша MAE. Результат не учитывает неизвестные реальные задержки публикации донорских измерений.", "",
              "| Целевая | Донор | Частный r: текущий target | Частный r: четыре контроля | N, четыре контроля |",
              "|---|---|---:|---:|---:|"]
    for target, donor in [(a, b) for pair in PAIRS for a, b in (pair, pair[::-1])]:
        rows = {r["controls"]: r for r in result["partial_correlations"] if r["target"] == target and r["period"] == "2024-2025"}
        lines.append(f"| {target} | {donor} | {rows[1]['partial_pearson']:.4f} | {rows[4]['partial_pearson']:.4f} | {rows[4]['n']} |")
    lines += ["", "## Устойчивость к фильтру качества", "", "| Пара | Представление | r, CS≥50 | r, CS≥70 | r, CS≥90 |",
              "|---|---|---:|---:|---:|"]
    for pair in ("PQ052_SO148", "IC437_JJ433"):
        for transform in names:
            rows = {r["minimum_cs"]: r for r in result["quality_sensitivity"] if r["pair"] == pair and r["transform"] == transform}
            lines.append(f"| {pair} | {names[transform]} | {rows[50]['pearson']:.4f} | {rows[70]['pearson']:.4f} | {rows[90]['pearson']:.4f} |")
    lines += ["", "При ужесточении CS меняются и качество, и набор временных интервалов; это проверка чувствительности, "
              "а не сравнение на строго одной выборке. Полные объёмы выборок находятся в cs_sensitivity.csv.", "",
              "## Изменения будущего эксперимента", "",
              "Протокол shared/protocol/nearby_pairs_2024_2025.json задаёт четыре направления: "
              "PQ052←SO148, SO148←PQ052, IC437←JJ433 и JJ433←IC437. У каждой цели ровно один донор из своей пары. "
              "Отдельный run_nearby_pairs.py готовит признаки, проверяет причинность, выполняет ежедневный расчёт "
              "с продолжением после остановки и сводный анализ. Старые прогнозы и движок не используются. "
              "Команды приведены в RUN_NEARBY_PAIRS.md. Подготовка не обучает модели.", "",
              "Диагностика выполнена на обоих годах по явному запросу. Эти данные уже просмотрены; последующий "
              "эксперимент нельзя описывать как полностью слепую подтверждающую проверку. Протокол фиксирует январь–март 2025 года для сравнения окон и апрель–декабрь "
              "2025 года для хронологической оценки; признаки используют заранее заданные исторические лаги. Существующие статьи, результаты и протокол завершённого "
              "эксперимента не заменены.", "", "## Воспроизведение и проверки", "",
              "Команда: `python shared/code/analyze_nearby_pairs.py`. Исходники находятся в "
              "`local_data/nearby_pairs_2024_2025/` и не добавляются в Git; SHA-256, относительные пути и источник "
              "зафиксированы в manifest.json. Выполнены проверки хешей, кодов станций, независимого подсчёта покрытия "
              "и средних, временной сетки, исключения 2026 года, отсутствия будущих данных в базовой линии "
              "и запрета разностей через пропуски. Прогнозные модели не импортируются и не запускаются.", "",
              "Первичный источник: [GIRO/DIDBase FastChar](https://giro.uml.edu/didbase/scaled.php). "
              "Архивные лучшие значения могут содержать поздние исправления и не доказывают доступность в реальном времени.", ""]
    for a, b in PAIRS:
        lines += [f"![Диагностика {a}–{b}]({a}_{b}_diagnostics.png)", ""]
    (out/"README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
