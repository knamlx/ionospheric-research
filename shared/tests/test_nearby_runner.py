"""Scientific boundary/recovery tests; estimators are stubs, no real training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import run_nearby_pairs as runner


class StubEstimator:
    """Check the training contract without learning any statistical model."""
    calls = 0

    def fit(self, x, y):
        type(self).calls += 1
        self.columns = list(x.columns)
        assert np.isfinite(x.to_numpy()).all()
        assert x.index.equals(y.index)
        return self

    def predict(self, x):
        assert list(x.columns) == self.columns
        assert np.isfinite(x.to_numpy()).all()
        return np.full(len(x), 6.)

    def get_params(self):
        return {"test_stub": True}


class NearbyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_json(runner.DEFAULT_CONFIG)
        self.index = pd.date_range("2024-12-01", "2025-01-04", freq="15min", tz="UTC", inclusive="left")
        phase = np.arange(len(self.index))/30
        self.panel = pd.DataFrame({"PQ052": 7+np.sin(phase), "SO148": 7+np.cos(phase),
                                   "IC437": 8+np.sin(phase+.2), "JJ433": 8+np.cos(phase+.2)}, index=self.index)
        self.origin = pd.Timestamp("2025-01-02 12:00", tz="UTC")

    def test_exact_plan_and_fixed_pairs(self):
        runner.validate_config(self.config)
        tasks = runner.generate_tasks(self.config)
        self.assertEqual(len(tasks), 10900)
        self.assertTrue(all(pd.Timestamp(t["day"]).year == 2025 for t in tasks))
        self.assertEqual(len({(t["target"], t["donor"]) for t in tasks}), 4)
        self.assertEqual(sum(t["phase"] == "exploratory" for t in tasks), 5400)
        self.assertEqual(sum(t["phase"] == "evaluation" for t in tasks), 5500)

    def test_future_hidden_history_and_unrelated_pair_cannot_change_features(self):
        for direction in self.config["directions"]:
            target, donor = direction["target"], direction["donors"][0]
            for gap in self.config["hidden_minutes"]:
                reference = runner.features(self.panel, target, donor, gap, self.config)
                changed = self.panel.copy()
                changed.loc[changed.index > self.origin-pd.Timedelta(minutes=gap), target] = 9999.
                changed.loc[changed.index > self.origin, donor] = 9999.
                for other in set(changed)-{target, donor}:
                    changed[other] = -9999.
                altered = runner.features(changed, target, donor, gap, self.config)
                for kind in (0, 1):
                    pd.testing.assert_series_equal(reference[kind].loc[self.origin], altered[kind].loc[self.origin])

    def test_continuous_outage_uses_one_pre_outage_anchor(self):
        start = pd.Timestamp("2025-01-02 06:00", tz="UTC")
        changed = self.panel.copy()
        changed.loc[(changed.index >= start) & (changed.index < start+pd.Timedelta(minutes=120)), "PQ052"] = 9999.
        for elapsed in (15, 30, 60, 120):
            origin = start+pd.Timedelta(minutes=elapsed-15)
            original = runner.features(self.panel, "PQ052", "SO148", elapsed, self.config)
            altered = runner.features(changed, "PQ052", "SO148", elapsed, self.config)
            for kind in (0, 1):
                pd.testing.assert_series_equal(original[kind].loc[origin], altered[kind].loc[origin])
            self.assertEqual(original[0].loc[origin, "foF2_state"], self.panel.loc[start-runner.STEP, "PQ052"])

    def test_long_missing_history_is_not_filled_indefinitely(self):
        changed = self.panel.copy()
        changed.loc["2024-12-30":"2025-01-03", "PQ052"] = np.nan
        local, _ = runner.features(changed, "PQ052", "SO148", 0, self.config)
        self.assertTrue(pd.isna(local.loc[self.origin, "last_available"]))

    def test_split_purges_future_labels_and_phase_end(self):
        task = {"day": "2025-01-02T00:00:00Z", "window_days": 7, "phase_end": "2025-01-03T00:00:00Z"}
        train, test = runner.split_indices(self.index, np.ones(len(self.index), dtype=bool), task)
        self.assertEqual(train.max()+2*runner.STEP, pd.Timestamp(task["day"]))
        self.assertEqual(test.max()+2*runner.STEP, pd.Timestamp(task["phase_end"]))
        self.assertEqual(len(train.intersection(test)), 0)

    def _stub_output(self, out):
        for folder in ("data", "checkpoints", "models", "results"):
            (out/folder).mkdir(exist_ok=True)
        local, remote = runner.features(self.panel, "PQ052", "SO148", 60, self.config)
        joblib.dump((local, remote), runner.feature_path(out, "PQ052", 60))
        eligible = (self.panel.PQ052.shift(-1).notna() & self.panel.SO148.notna() & local.last_available.notna()).to_numpy()
        runner.STATE.clear()
        runner.STATE.update(out=out, config=self.config, fingerprint="test-only-not-a-scientific-run", panel=self.panel,
                            eligibility={"PQ052": eligible}, cache_key=None)
        return {"id": "stub_checkpoint", "phase": "exploratory", "target": "PQ052", "donor": "SO148",
                "day": "2025-01-02T00:00:00Z", "window_days": 7, "gap_minutes": 60,
                "phase_end": "2025-01-04T00:00:00Z"}

    def test_task_serialization_train_only_imputation_and_resume(self):
        StubEstimator.calls = 0
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            task = self._stub_output(out)
            with patch.object(runner, "make_estimator", return_value=StubEstimator()):
                result = runner.run_task(task)
                self.assertEqual(result["status"], "completed", result)
                count = StubEstimator.calls
                resumed = runner.run_task(task)
                self.assertTrue(resumed["resumed"])
                self.assertEqual(StubEstimator.calls, count)
            self.assertEqual(count, 10)  # stub calls, never actual estimators
            record = runner.load_json(out/"checkpoints/stub_checkpoint.json")
            train, _ = runner.split_indices(self.index, runner.STATE["eligibility"]["PQ052"], task)
            local = runner.STATE["frames"][0]
            for column, median in record["medians"]["local"].items():
                self.assertAlmostEqual(median, float(local.loc[train, column].median()))
            predictions = pd.read_csv(out/"checkpoints/stub_checkpoint.csv")
            self.assertTrue({"delay15__CatBoost", "delay30__CatBoost", "interstation__CatBoost", "local__CatBoost"} <= set(predictions))
            self.assertTrue(predictions.shared_delays.all())
            (out/"checkpoints/stub_checkpoint.csv").write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "corrupted"):
                runner.checkpoint(out, task, runner.STATE["fingerprint"])

    def test_failed_task_has_no_committed_marker_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            task = self._stub_output(out)
            with patch.object(runner, "make_estimator", side_effect=RuntimeError("injected test failure")):
                self.assertEqual(runner.run_task(task)["status"], "failed")
            self.assertFalse((out/"checkpoints/stub_checkpoint.json").exists())
            self.assertTrue((out/"checkpoints/stub_checkpoint.error.json").exists())
            with patch.object(runner, "make_estimator", return_value=StubEstimator()):
                self.assertEqual(runner.run_task(task)["status"], "completed")
            self.assertFalse((out/"checkpoints/stub_checkpoint.error.json").exists())

    def test_last_day_replay_bundles_are_hashed_and_checked(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            task = self._stub_output(out)
            task.update(window_days=21, phase_end="2025-01-03T00:00:00Z")
            with patch.object(runner, "make_estimator", return_value=StubEstimator()):
                result = runner.run_task(task)
            self.assertEqual(result["status"], "completed", result)
            record = runner.checkpoint(out, task, runner.STATE["fingerprint"])
            self.assertEqual(len(record["model_files"]), 10)
            replay = out/next(iter(record["model_files"]))
            bundle = joblib.load(replay)
            self.assertEqual(bundle["task"], task)
            replay.write_bytes(b"corrupted-test-bundle")
            with self.assertRaisesRegex(RuntimeError, "replay model"):
                runner.checkpoint(out, task, runner.STATE["fingerprint"])

    def test_summary_matches_windows_and_writes_all_scenarios_without_real_models(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            first = self._stub_output(out)
            second = {**first, "id": "stub_second_window", "window_days": 21}
            with patch.object(runner, "make_estimator", return_value=StubEstimator()):
                self.assertEqual(runner.run_task(first)["status"], "completed")
                self.assertEqual(runner.run_task(second)["status"], "completed")
            # An intentionally missing timestamp in one window must be excluded from both.
            path = out/"checkpoints/stub_second_window.csv"
            predictions = pd.read_csv(path).iloc[1:]
            predictions.to_csv(path, index=False)
            record = runner.load_json(path.with_suffix(".json"))
            record["predictions_sha256"] = runner.sha(path)
            runner.atomic_json(path.with_suffix(".json"), record)
            config = json.loads(json.dumps(self.config))
            config["phases"] = [{"name": "exploratory", "start_utc": first["day"],
                                  "end_exclusive_utc": first["phase_end"], "windows_days": [7, 21]}]
            config["bootstrap_repetitions"] = 200
            config["primary_comparisons"] = {"stub": {"phase": "exploratory", "model": "LinearRegression",
                                                      "gap_minutes": 60, "window_days": 7, "scenario": "standard"}}
            runner.atomic_json(out/"tasks.json", [first, second])
            self.panel.to_pickle(out/"data/panel.pkl")
            joblib.dump(runner.STATE["eligibility"], out/"data/eligibility.joblib")
            with patch.object(runner, "verify_prepared", return_value=(config, {"fingerprint": runner.STATE["fingerprint"]})):
                runner.summarize(out, runner.DEFAULT_CONFIG)
            metrics = pd.read_csv(out/"results/metrics.csv")
            standard = metrics.loc[metrics.scenario.eq("standard")]
            self.assertEqual(standard.n.nunique(), 1)
            self.assertEqual(int(standard.n.iloc[0]), len(predictions))
            self.assertEqual(set(metrics.scenario), {"standard", "delay15", "delay30", "continuous_outage"})
            self.assertTrue(metrics.paired_mae_difference.eq(0).all())
            self.assertEqual(len(pd.read_csv(out/"results/primary_contrasts.csv")), 1)
            self.assertEqual(runner.load_json(out/"results/summary.json")["status"], "complete")

    def test_prepare_guard_rejects_real_fit(self):
        with runner.training_prohibited():
            estimator = runner.make_estimator("LinearRegression", self.config)
            with self.assertRaisesRegex(RuntimeError, "Training forbidden"):
                estimator.fit([[1.], [2.]], [1., 2.])

    def test_calendar_block_bootstrap_constant_paired_difference(self):
        loss = pd.DataFrame({"n": [2, 0, 2, 2, 0, 2, 2], "difference_sum": [2, 0, 2, 2, 0, 2, 2]})
        self.assertEqual(runner.bootstrap_difference(loss, 200, 3, 4), (1., 1., 1., 1.))

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as name:
            with runner.output_lock(name):
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    with runner.output_lock(name):
                        pass


if __name__ == "__main__":
    unittest.main()
