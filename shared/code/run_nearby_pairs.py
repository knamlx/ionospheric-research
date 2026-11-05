"""Prepare, resume and summarize the fixed nearby-pair experiment.

`prepare` never fits an estimator. `run` is the explicit training entry point.
All artifacts stay in an isolated local_data directory; historical engines are
neither imported nor used as checkpoint sources.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "shared/protocol/nearby_pairs_2024_2025.json"
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "joblib", "threadpoolctl", "matplotlib")
STEP = pd.Timedelta(minutes=15)
STATE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, default=str, allow_nan=False)+"\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_classes():
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    return dict(LinearRegression=LinearRegression, ElasticNet=ElasticNet,
                RandomForest=RandomForestRegressor, XGBoost=XGBRegressor, CatBoost=CatBoostRegressor)


def make_estimator(name, config):
    params = dict(config["model_params"].get(name, {}))
    if name == "CatBoost":
        params["allow_writing_files"] = False
    return model_classes()[name](**params)


@contextmanager
def training_prohibited():
    """Turn accidental learning during preparation into a hard error."""
    from sklearn.impute import SimpleImputer
    with ExitStack() as stack:
        for cls in model_classes().values():
            stack.enter_context(patch.object(cls, "fit", side_effect=RuntimeError("Training forbidden during prepare")))
        stack.enter_context(patch.object(SimpleImputer, "fit", side_effect=RuntimeError("Imputer fit forbidden during prepare")))
        yield


def environment(config):
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    effective = {name: make_estimator(name, config).get_params() for name in config["models"]}
    return json_safe({"python": platform.python_version(), "packages": versions, "model_parameters": effective})


@contextmanager
def output_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/".runner.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another preparation/training/summary process is already using this output directory") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_config(config):
    assert config["schema_version"] == 1
    assert config["parameter_search"] is False
    assert config["dataset_step_minutes"] == config["forecast_horizon_minutes"] == 15
    assert set(config["hidden_minutes"]) == {0, 15, 30, 60, 120}
    assert config["donor_delay_minutes"] == [0, 15, 30]
    assert config["models"] == ["LinearRegression", "ElasticNet", "RandomForest", "XGBoost", "CatBoost"]
    assert pd.Timestamp(config["period_start_utc"]) == pd.Timestamp("2024-01-01", tz="UTC")
    assert pd.Timestamp(config["period_end_exclusive_utc"]) == pd.Timestamp("2026-01-01", tz="UTC")
    expected = {(a, b) for pair in config["pairs"] for a, b in (pair, pair[::-1])}
    assert expected == {(d["target"], d["donors"][0]) for d in config["directions"]}
    assert all(len(d["donors"]) == 1 for d in config["directions"])
    previous = None
    for phase in config["phases"]:
        a, b = pd.Timestamp(phase["start_utc"]), pd.Timestamp(phase["end_exclusive_utc"])
        assert a < b <= pd.Timestamp(config["period_end_exclusive_utc"])
        assert a == a.normalize() and b == b.normalize()
        if previous is not None:
            assert previous == a, "Phases must be contiguous and nonoverlapping"
        previous = b


def features(panel, target, donor, gap, config):
    """A single fixed donor, only completed local history before the hidden gap."""
    if target == donor or {target, donor} not in [set(p) for p in config["pairs"]]:
        raise ValueError("Target and donor must belong to the same declared pair")
    index = panel.index
    x = pd.DataFrame(index=index)
    for name, phase in (("hour", index.hour/24), ("month", index.month/12), ("doy", index.dayofyear/366)):
        x[name+"_sin"] = np.sin(2*np.pi*phase)
        x[name+"_cos"] = np.cos(2*np.pi*phase)
    visible = panel[target].shift(gap//15)
    x["foF2"] = visible
    x["foF2_state"] = visible
    for lag in config["features"]["lags"]:
        x["foF2_lag_"+lag] = visible.shift(int(pd.Timedelta(lag)/STEP))
    for lag in config["features"]["diff_lags"]:
        x["foF2_diff_"+lag] = visible-visible.shift(int(pd.Timedelta(lag)/STEP))
    stamp = pd.Series(index, index=index).where(panel[target].notna()).shift(gap//15).ffill()
    age = (pd.Series(index, index=index)-stamp).dt.total_seconds()/60
    fresh = age.le(config["features"]["maximum_local_age_minutes"])
    x["last_available"] = visible.ffill().where(fresh)
    x["age_minutes"] = age.where(fresh)
    remote = x.copy()
    remote[donor+"_state"] = panel[donor]
    for lag in config["features"]["lags"]:
        remote[donor+"_lag_"+lag] = panel[donor].shift(int(pd.Timedelta(lag)/STEP))
    assert not any(c.startswith(d+"_") for d in panel if d not in (target, donor) for c in remote)
    return x, remote


def feature_path(out, target, gap):
    return Path(out)/"data"/f"features_{target}_g{gap}.joblib"


def generate_tasks(config):
    tasks = []
    for phase in config["phases"]:
        days = pd.date_range(phase["start_utc"], phase["end_exclusive_utc"], freq="D", inclusive="left")
        for direction in config["directions"]:
            target, donor = direction["target"], direction["donors"][0]
            # Group tasks so a worker can reuse the most recently loaded feature frame.
            for gap in config["hidden_minutes"]:
                for window in phase["windows_days"]:
                    for day in days:
                        name = f"{phase['name']}_{target}_from_{donor}_{day:%Y%m%d}_w{window}_g{gap}"
                        tasks.append(dict(id=name, phase=phase["name"], target=target, donor=donor,
                                          day=str(day), window_days=window, gap_minutes=gap,
                                          phase_end=phase["end_exclusive_utc"]))
    assert len(tasks) == len({t["id"] for t in tasks})
    return tasks


def split_indices(index, eligible, task):
    day = pd.Timestamp(task["day"])
    lo = index.searchsorted(day-pd.Timedelta(days=task["window_days"]))
    hi = index.searchsorted(day-2*STEP, side="right")
    train = index[lo:hi][eligible[lo:hi]]
    lo = index.searchsorted(day)
    hi = index.searchsorted(min(day+pd.Timedelta(days=1), pd.Timestamp(task["phase_end"])))
    test = index[lo:hi][eligible[lo:hi]]
    test = test[test+2*STEP <= pd.Timestamp(task["phase_end"])]
    return train, test


def causal_checks(panel, config):
    origin = pd.Timestamp("2025-05-15 12:00", tz="UTC")
    checks = []
    for direction in config["directions"]:
        target, donor = direction["target"], direction["donors"][0]
        for gap in config["hidden_minutes"]:
            reference = features(panel, target, donor, gap, config)
            changed = panel.copy()
            changed.loc[changed.index > origin-pd.Timedelta(minutes=gap), target] = 12345.
            changed.loc[changed.index > origin, donor] = 12345.
            for other in set(panel.columns)-{target, donor}:
                changed[other] = -12345.
            altered = features(changed, target, donor, gap, config)
            for kind in (0, 1):
                pd.testing.assert_series_equal(reference[kind].loc[origin], altered[kind].loc[origin])
            checks.append(dict(target=target, donor=donor, gap_minutes=gap, passed=True))
        start = pd.Timestamp("2025-05-15 06:00", tz="UTC")
        changed = panel.copy()
        changed.loc[(changed.index >= start) & (changed.index < start+pd.Timedelta(minutes=120)), target] = 12345.
        for elapsed in config["continuous_outage"]["checkpoint_minutes"]:
            at = start+pd.Timedelta(minutes=elapsed-15)
            reference = features(panel, target, donor, elapsed, config)
            altered = features(changed, target, donor, elapsed, config)
            for kind in (0, 1):
                pd.testing.assert_series_equal(reference[kind].loc[at], altered[kind].loc[at])
            checks.append(dict(target=target, donor=donor, outage_elapsed_minutes=elapsed, passed=True))
    return checks


def preparation_identity(config_path, config, sources, env):
    identity = {"engine_sha256": sha(__file__), "protocol_sha256": sha(config_path),
                "diagnostic_parser_sha256": sha(ROOT/"shared/code/analyze_nearby_pairs.py"),
                "source_manifest_sha256": sha(ROOT/config["data_directory"]/"manifest.json"),
                "source_files": [{k: s[k] for k in ("file", "sha256", "bytes")} for s in sources["sources"]],
                "environment": env, "experiment_id": config["experiment_id"]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return identity, fingerprint


def prepare(config_path, out):
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    config = load_json(config_path)
    validate_config(config)
    data_root = ROOT/config["data_directory"]
    sources = load_json(data_root/"manifest.json")
    if out == data_root.resolve() or data_root.resolve() in out.parents:
        raise ValueError("Output must be separate from the archived input dataset")
    if out in [(ROOT/"local_data/experiment").resolve(), (ROOT/"local_data/baseline").resolve()]:
        raise ValueError("Historical experiment directories are protected")
    env = environment(config)
    identity, fingerprint = preparation_identity(config_path, config, sources, env)
    marker = out/"run_manifest.json"
    if marker.exists():
        if load_json(marker)["fingerprint"] != fingerprint:
            raise RuntimeError("Prepared experiment differs from current code/configuration/environment; use a new output directory")
        verify_prepared(out, config_path, verify_raw=True)
        print("Prepared inputs already verified; no training started.", flush=True)
        return
    if out.exists() and any(p.name != ".runner.lock" for p in out.iterdir()) and not (out/"preparing.json").exists():
        raise RuntimeError("Refusing a nonempty output directory without this experiment's preparation marker")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"preparing.json").exists() and load_json(out/"preparing.json")["fingerprint"] != fingerprint:
        raise RuntimeError("An interrupted preparation belongs to different inputs; use a new directory")
    atomic_json(out/"preparing.json", {"fingerprint": fingerprint})
    for directory in ("data", "checkpoints", "models", "results", "source"):
        (out/directory).mkdir(exist_ok=True)
    # Reuse only the independent parser, never a forecasting engine.
    import analyze_nearby_pairs as diagnostic
    frames, audits = diagnostic.load_observations(data_root, sources)
    panel = diagnostic.panel_for_cs(frames, config["data_filter"]["minimum_cs"])
    assert len(panel) == 70176 and panel.index[-1] < pd.Timestamp("2026-01-01", tz="UTC")
    for station in panel:
        assert int(panel[station].notna().sum()) == sources["expected_valid_bins"][station]
    panel.to_pickle(out/"data/panel.pkl")
    atomic_json(out/"protocol.json", config)
    atomic_json(out/"data/sources.json", sources)
    checks = causal_checks(panel, config)
    eligibility = {}
    artifacts = {"data/panel.pkl": sha(out/"data/panel.pkl"), "protocol.json": sha(out/"protocol.json")}
    for direction in config["directions"]:
        target, donor = direction["target"], direction["donors"][0]
        available = pd.Series(True, index=panel.index)
        for gap in config["hidden_minutes"]:
            local, remote = features(panel, target, donor, gap, config)
            path = feature_path(out, target, gap)
            joblib.dump((local, remote), path, compress=3)
            artifacts[path.relative_to(out).as_posix()] = sha(path)
            available &= local.last_available.notna()
        eligibility[target] = (available & panel[target].shift(-1).notna() & panel[donor].notna()).to_numpy()
        print(f"Prepared {target} <- {donor}: all 5 hidden-history scenarios", flush=True)
    joblib.dump(eligibility, out/"data/eligibility.joblib", compress=3)
    artifacts["data/eligibility.joblib"] = sha(out/"data/eligibility.joblib")
    tasks = generate_tasks(config)
    planning = []
    for task in tasks:
        train, test = split_indices(panel.index, eligibility[task["target"]], task)
        if len(train):
            assert (train+2*STEP).max() <= pd.Timestamp(task["day"])
        assert not len(train.intersection(test))
        planning.append({**task, "n_train": len(train), "n_test": len(test),
                         "row_counts_sufficient": len(train) >= config["minimum_train_rows"] and len(test) >= config["minimum_eval_rows"]})
    atomic_json(out/"tasks.json", tasks)
    artifacts["tasks.json"] = sha(out/"tasks.json")
    pd.DataFrame(planning).to_csv(out/"task_plan.csv", index=False, lineterminator="\n")
    row_sufficient = sum(row["row_counts_sufficient"] for row in planning)
    ready = {"status": "ready", "forecast_training_started": False, "estimator_fit_calls": 0,
             "tasks": len(tasks), "tasks_with_sufficient_row_counts": row_sufficient,
             "potentially_skipped_for_row_counts": len(tasks)-row_sufficient,
             "maximum_estimator_fits": len(tasks)*len(config["models"])*2,
             "causality_checks": len(checks), "all_causality_checks_passed": all(c["passed"] for c in checks),
             "directions": config["directions"], "environment": env,
             "notes": ["No estimator or imputer was fitted during preparation", "Actual future skips may also include constant training labels"]}
    atomic_json(out/"readiness.json", ready)
    atomic_json(out/"data/causality_checks.json", checks)
    pd.DataFrame(audits).to_csv(out/"data/source_quality_audit.csv", index=False, lineterminator="\n")
    # Archive exact runnable source for audit; never execute a mutable historical script.
    for source in (Path(__file__), ROOT/"shared/code/analyze_nearby_pairs.py", config_path):
        (out/"source"/source.name).write_bytes(source.read_bytes())
    (out/"runtime_requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items())+"\n", encoding="utf-8", newline="\n")
    atomic_json(marker, {"fingerprint": fingerprint, "identity": identity, "artifacts": artifacts,
                         "created_utc": datetime.now(timezone.utc).isoformat(), "input_root": str(data_root.resolve())})
    print(json.dumps({k: v for k, v in ready.items() if k != "environment"}, ensure_ascii=False), flush=True)


def verify_prepared(out, config_path, verify_raw=True):
    out = Path(out)
    marker = load_json(out/"run_manifest.json")
    config = load_json(config_path)
    sources = load_json(ROOT/config["data_directory"]/"manifest.json")
    _, expected = preparation_identity(config_path, config, sources, environment(config))
    if marker["fingerprint"] != expected:
        raise RuntimeError("Code, protocol, raw manifest or environment changed since preparation")
    for relative, digest in marker["artifacts"].items():
        if not (out/relative).exists() or sha(out/relative) != digest:
            raise RuntimeError("Prepared artifact missing or modified: "+relative)
    if verify_raw:
        for entry in sources["sources"]:
            if sha(ROOT/config["data_directory"]/entry["file"]) != entry["sha256"]:
                raise RuntimeError("Raw source modified: "+entry["file"])
    return config, marker


def initialize_worker(out):
    out = Path(out)
    STATE.clear()
    STATE.update(out=out, config=load_json(out/"protocol.json"),
                 fingerprint=load_json(out/"run_manifest.json")["fingerprint"],
                 panel=pd.read_pickle(out/"data/panel.pkl"),
                 eligibility=joblib.load(out/"data/eligibility.joblib"), cache_key=None)


def checkpoint(out, task, fingerprint):
    path = Path(out)/"checkpoints"/(task["id"]+".json")
    if not path.exists():
        return None
    result = load_json(path)
    if result["fingerprint"] != fingerprint or result["task"] != task:
        raise RuntimeError("Checkpoint identity mismatch: "+task["id"])
    if result["status"] == "completed":
        data = path.with_suffix(".csv")
        if not data.exists() or sha(data) != result["predictions_sha256"]:
            raise RuntimeError("Checkpoint predictions missing or corrupted: "+task["id"])
        for relative, digest in result.get("model_files", {}).items():
            if not (Path(out)/relative).exists() or sha(Path(out)/relative) != digest:
                raise RuntimeError("Saved replay model missing or corrupted: "+relative)
    elif result["status"] != "skipped":
        raise RuntimeError("Invalid checkpoint status: "+task["id"])
    return result


def run_task(task):
    out, config, panel = STATE["out"], STATE["config"], STATE["panel"]
    existing = checkpoint(out, task, STATE["fingerprint"])
    if existing:
        return {"id": task["id"], "status": existing["status"], "resumed": True}
    started = time.perf_counter()
    record = {"task": task, "fingerprint": STATE["fingerprint"]}
    try:
        target, donor, gap = task["target"], task["donor"], task["gap_minutes"]
        key = (target, gap)
        if STATE["cache_key"] != key:
            STATE["frames"] = joblib.load(feature_path(out, target, gap))
            STATE["cache_key"] = key
        local, multi = STATE["frames"]
        train, test = split_indices(panel.index, STATE["eligibility"][target], task)
        y = panel[target].shift(-1)
        record.update(n_train=len(train), n_test=len(test))
        reason = ("insufficient_train" if len(train) < config["minimum_train_rows"] else
                  "insufficient_test" if len(test) < config["minimum_eval_rows"] else
                  "constant_train_target" if y.loc[train].nunique() < 2 else None)
        if reason:
            record.update(status="skipped", reason=reason, seconds=time.perf_counter()-started)
            atomic_json(out/"checkpoints"/(task["id"]+".json"), record)
            return {"id": task["id"], "status": "skipped", "resumed": False}
        assert (train+2*STEP).max() <= pd.Timestamp(task["day"])
        assert (test+2*STEP).max() <= pd.Timestamp(task["phase_end"])
        assert not len(train.intersection(test))
        record.update(train_first=str(train.min()), train_last=str(train.max()),
                      last_training_target_end=str((train+2*STEP).max()), features={}, medians={}, warnings=[], model_files={})
        shared_delays = pd.Series(True, index=panel.index)
        for minutes in config["donor_delay_minutes"]:
            shared_delays &= panel[donor].shift(minutes//15).notna()
        pred = pd.DataFrame({"time_utc": test, "forecast_origin_utc": test+STEP,
                             "target_end_utc": test+2*STEP, "actual": y.loc[test].to_numpy(),
                             "persistence": local.loc[test, "last_available"].to_numpy(),
                             "shared_delays": shared_delays.loc[test].to_numpy()})
        from sklearn.impute import SimpleImputer
        from threadpoolctl import threadpool_limits
        import warnings
        for kind, frame in (("local", local), ("interstation", multi)):
            columns = frame.columns[frame.loc[train].notna().mean().ge(config["features"]["minimum_train_coverage"])].tolist()
            if kind == "interstation":
                assert donor+"_state" in columns
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            xt = pd.DataFrame(imp.fit_transform(frame.loc[train, columns]), columns=columns, index=train)
            xe = pd.DataFrame(imp.transform(frame.loc[test, columns]), columns=columns, index=test)
            record["features"][kind] = columns
            record["medians"][kind] = dict(zip(columns, imp.statistics_.tolist()))
            delayed = {}
            if kind == "interstation":
                for minutes in config["donor_delay_minutes"]:
                    if minutes == 0:
                        continue
                    changed = frame.loc[test, columns].copy()
                    for column in columns:
                        if column.startswith(donor+"_"):
                            changed[column] = frame[column].shift(minutes//15).loc[test]
                    delayed[minutes] = pd.DataFrame(imp.transform(changed), columns=columns, index=test)
            for model_name in config["models"]:
                estimator = make_estimator(model_name, config)
                with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
                    warnings.simplefilter("always")
                    estimator.fit(xt, y.loc[train])
                    pred[kind+"__"+model_name] = np.asarray(estimator.predict(xe), dtype=float)
                    for minutes, xd in delayed.items():
                        pred[f"delay{minutes}__{model_name}"] = np.asarray(estimator.predict(xd), dtype=float)
                for warning in captured:
                    record["warnings"].append({"kind": kind, "model": model_name, "message": str(warning.message)})
                # Keep replayable models only on the last day of each phase, fixed 21-day window.
                if task["window_days"] == 21 and pd.Timestamp(task["day"])+pd.Timedelta(days=1) == pd.Timestamp(task["phase_end"]):
                    bundle = dict(columns=columns, imputer=imp, estimator=estimator,
                                  train_index=train, test_index=test, task=task, parameters=estimator.get_params())
                    model_path = out/"models"/(task["id"]+"_"+kind+"_"+model_name+".joblib")
                    temporary_model = model_path.with_suffix(".tmp")
                    joblib.dump(bundle, temporary_model, compress=3)
                    temporary_model.replace(model_path)
                    record["model_files"][model_path.relative_to(out).as_posix()] = sha(model_path)
        prediction_columns = [c for c in pred if "__" in c]
        assert np.isfinite(pred[prediction_columns].to_numpy()).all()
        destination = out/"checkpoints"/(task["id"]+".csv")
        temporary = destination.with_suffix(".csv.tmp")
        pred.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        temporary.replace(destination)
        record.update(status="completed", seconds=time.perf_counter()-started,
                      predictions_sha256=sha(destination), prediction_columns=prediction_columns)
        atomic_json(destination.with_suffix(".json"), record)
        error = out/"checkpoints"/(task["id"]+".error.json")
        if error.exists():
            error.unlink()
        return {"id": task["id"], "status": "completed", "resumed": False}
    except Exception:
        record.update(status="failed", error=traceback.format_exc(), seconds=time.perf_counter()-started)
        atomic_json(out/"checkpoints"/(task["id"]+".error.json"), record)
        return {"id": task["id"], "status": "failed", "error": record["error"], "resumed": False}


def run(out, config_path, workers):
    config, marker = verify_prepared(out, config_path)
    tasks = load_json(out/"tasks.json")
    cached, pending = [], []
    for task in tasks:
        found = checkpoint(out, task, marker["fingerprint"])
        (cached if found else pending).append(found if found else task)
    print(f"Verified {len(cached)} finished checkpoints; remaining {len(pending)} / {len(tasks)} tasks; workers={workers}", flush=True)
    started = time.perf_counter()
    counts = Counter(r["status"] for r in cached)
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(str(out),)) as pool:
            futures = [pool.submit(run_task, task) for task in pending]
            for future in as_completed(futures):
                result = future.result()
                counts[result["status"]] += 1
                if result["status"] == "failed":
                    failures.append(result)
                progress = {"done": sum(counts.values()), "total": len(tasks), "counts": dict(counts),
                            "latest": result["id"], "session_seconds": time.perf_counter()-started,
                            "resumed_checkpoints": len(cached)}
                atomic_json(out/"progress.json", progress)
                if progress["done"] % 10 == 0 or result["status"] == "failed":
                    print(json.dumps(progress), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} tasks failed. Inspect *.error.json and rerun the same command to retry them.")
    atomic_json(out/"training_finished.json", {"fingerprint": marker["fingerprint"], "counts": dict(counts),
                                              "tasks": len(tasks), "finished_utc": datetime.now(timezone.utc).isoformat()})
    summarize(out, config_path)


def bootstrap_difference(daily, repetitions, block_days, family):
    if daily["n"].sum() == 0:
        return None, None, None, None
    values = daily[["n", "difference_sum"]].to_numpy()
    count = len(values)
    rng = np.random.default_rng(20261102)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, count, size=int(np.ceil(count/block_days)))
        ids = ((starts[:, None]+np.arange(block_days)) % count).ravel()[:count]
        n, difference = values[ids].sum(axis=0)
        if n:
            draws.append(difference/n)
    if not draws:
        return None, None, None, None
    quantiles = np.quantile(draws, [.025, .975, .025/family, 1-.025/family])
    return tuple(float(v) for v in quantiles)


def summarize(out, config_path):
    config, marker = verify_prepared(out, config_path)
    tasks = load_json(out/"tasks.json")
    # Validate every marker/file before reading any predictions for scoring.
    records = []
    for task in tasks:
        record = checkpoint(out, task, marker["fingerprint"])
        if record is None:
            raise RuntimeError("Training is incomplete; summary requires all planned checkpoints")
        records.append(record)
    groups = {}
    for record in records:
        task = record["task"]
        key = (task["phase"], task["target"], task["donor"], task["gap_minutes"])
        groups.setdefault(key, {}).setdefault(task["window_days"], []).append(record)
    panel = pd.read_pickle(out/"data/panel.pkl")
    eligibility = joblib.load(out/"data/eligibility.joblib")
    rows, daily_rows, coverage_rows = [], [], []
    for (phase, target, donor, gap), windows in groups.items():
        phase_config = next(p for p in config["phases"] if p["name"] == phase)
        phase_mask = ((panel.index >= pd.Timestamp(phase_config["start_utc"]))
                      & (panel.index+2*STEP <= pd.Timestamp(phase_config["end_exclusive_utc"])))
        target_observed = phase_mask & panel[target].shift(-1).notna().to_numpy()
        reference_count = int(target_observed.sum())
        eligible_count = int((phase_mask & eligibility[target]).sum())
        loaded = {}
        for window, items in windows.items():
            parts = [pd.read_csv(out/"checkpoints"/(r["task"]["id"]+".csv"), parse_dates=["time_utc"])
                     for r in items if r["status"] == "completed"]
            loaded[window] = pd.concat(parts, ignore_index=True).set_index("time_utc").sort_index() if parts else pd.DataFrame()
            if len(loaded[window]) and loaded[window].index.has_duplicates:
                raise RuntimeError("Duplicate forecast origins in summary")
        # Missing/skipped windows contribute the empty set, never an easier comparison.
        shared = None
        for window in phase_config["windows_days"]:
            index = loaded[window].index if len(loaded[window]) else pd.DatetimeIndex([], tz="UTC")
            shared = index if shared is None else shared.intersection(index)
        for window, frame in loaded.items():
            coverage_rows.append(dict(phase=phase, target=target, donor=donor, gap_minutes=gap,
                                      window_days=window, scheduled_origins=int(phase_mask.sum()),
                                      target_observed_origins=reference_count,
                                      eligible_history_and_donor_origins=eligible_count,
                                      unavailable_due_to_missing_inputs=reference_count-eligible_count,
                                      eligible_but_no_fit=eligible_count-len(frame),
                                      available_predictions=len(frame), common_window_predictions=len(shared),
                                      forecast_coverage_of_observed_target_pct=100*len(frame)/reference_count if reference_count else None,
                                      common_window_coverage_of_observed_target_pct=100*len(shared)/reference_count if reference_count else None))
            if frame.empty or not len(shared):
                continue
            frame = frame.loc[shared]
            for scenario in ("standard", "delay15", "delay30", "continuous_outage"):
                selected = frame
                remote_prefix = "interstation" if scenario in ("standard", "continuous_outage") else scenario
                if scenario in ("delay15", "delay30"):
                    selected = frame.loc[frame["shared_delays"].astype(str).str.lower().eq("true")]
                if scenario == "continuous_outage":
                    if gap not in config["continuous_outage"]["checkpoint_minutes"]:
                        continue
                    episode_start = frame.index-pd.Timedelta(minutes=gap-15)
                    starts = set(config["continuous_outage"]["starts_utc"])
                    selected = frame.loc[[stamp.strftime("%H:%M") in starts for stamp in episode_start]]
                if selected.empty:
                    continue
                for name in config["models"]:
                    actual = selected["actual"].to_numpy()
                    local = np.abs(selected["local__"+name].to_numpy()-actual)
                    remote = np.abs(selected[remote_prefix+"__"+name].to_numpy()-actual)
                    persistence = np.abs(selected["persistence"].to_numpy()-actual)
                    loss = pd.DataFrame({"n": 1, "local_sum": local, "remote_sum": remote,
                                         "difference_sum": local-remote}, index=selected.index).resample("D").sum()
                    days = pd.date_range(phase_config["start_utc"], phase_config["end_exclusive_utc"], freq="D", inclusive="left")
                    loss = loss.reindex(days, fill_value=0)
                    low, high, simultaneous_low, simultaneous_high = bootstrap_difference(
                        loss, config["bootstrap_repetitions"], config["bootstrap_block_calendar_days"], config["multiplicity_family_size"])
                    metadata = dict(phase=phase, target=target, donor=donor, gap_minutes=gap, window_days=window,
                                    scenario=scenario, model=name)
                    rows.append({**metadata, "n": len(selected), "local_mae": float(local.mean()),
                                 "interstation_mae": float(remote.mean()), "persistence_mae": float(persistence.mean()),
                                 "mae_reduction_percent": float(100*(local.mean()-remote.mean())/local.mean()) if local.mean() else None,
                                 "paired_mae_difference": float((local-remote).mean()), "ci95_low": low, "ci95_high": high,
                                 "ci_family4_low": simultaneous_low, "ci_family4_high": simultaneous_high})
                    daily_rows.extend({**metadata, "date_utc": str(day), **row} for day, row in loss.to_dict("index").items())
    primary = []
    for row in rows:
        for article, requirements in config["primary_comparisons"].items():
            if all(row[key] == value for key, value in requirements.items()):
                primary.append({"article": article, **row})
    for filename, values in (("metrics.csv", rows), ("primary_contrasts.csv", primary),
                             ("daily_losses.csv", daily_rows), ("forecast_coverage.csv", coverage_rows)):
        pd.DataFrame(values).to_csv(out/"results"/filename, index=False, float_format="%.12g", lineterminator="\n")
    atomic_json(out/"results/summary.json", {"status": "complete", "groups": len(rows), "protocol": config["experiment_id"],
                                            "fingerprint": marker["fingerprint"], "evaluation_disclosure": config["evaluation_disclosure"],
                                            "positive_difference_means": "Local MAE minus interstation MAE; positive favors donor",
                                            "family4_intervals": "Bonferroni percentile intervals over four target directions within a fixed scenario/model/window/gap"})
    print(f"Summary complete: {out/'results/metrics.csv'}", flush=True)


def status(out):
    out = Path(out)
    if not (out/"run_manifest.json").exists():
        print(json.dumps({"status": "not_prepared", "output": str(out)}))
        return
    marker = load_json(out/"run_manifest.json")
    tasks = load_json(out/"tasks.json")
    counts = Counter()
    for task in tasks:
        result = checkpoint(out, task, marker["fingerprint"])
        counts[result["status"] if result else "pending"] += 1
    print(json.dumps({"status": "prepared" if counts["pending"] == len(tasks) else "in_progress" if counts["pending"] else "training_finished",
                      "output": str(out), "tasks": len(tasks), "counts": dict(counts),
                      "summary_complete": (out/"results/summary.json").exists()}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "status", "summary"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = load_json(args.config)
    out = args.output.resolve() if args.output else ROOT/config["output_directory"]
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.action == "status":
        status(out)
        return
    data_root = (ROOT/config["data_directory"]).resolve()
    if out.resolve() == data_root or data_root in out.resolve().parents:
        raise ValueError("Output cannot be inside the archived input directory")
    if out.resolve() in [(ROOT/"local_data/experiment").resolve(), (ROOT/"local_data/baseline").resolve()]:
        raise ValueError("Historical experiment directories are protected")
    with output_lock(out):
        if args.action == "prepare":
            with training_prohibited():
                prepare(args.config, out)
        elif args.action == "run":
            if not (out/"run_manifest.json").exists():
                with training_prohibited():
                    prepare(args.config, out)
            run(out, args.config, args.workers)
        else:
            summarize(out, args.config)


if __name__ == "__main__":
    main()
