import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis import position_gain_sweep as sweep


class _FakeLog:
    arms = ("right", "left")
    settled = True
    settle_duration = 1.25
    valid = True
    warning_reasons = ()
    evaluation_mask = np.array([False, True, True])
    arm_data = {
        "right": {
            "e_pos": np.array([[99, 99, 99], [3, 4, 0], [0, 0, 12]]) / 1000,
            "e_v": np.array([[99, 99, 99], [3, 4, 0], [0, 0, 12]]) / 10,
            "qdot_measured": np.array([[99] * 7, [1, -2, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 3, 0, 0, 0]]),
        },
        "left": {
            "e_pos": np.array([[99, 99, 99], [0, 6, 8], [0, 0, 0]]) / 1000,
            "e_v": np.array([[99, 99, 99], [0, 0.6, 0.8], [0, 0, 0]]),
            "qdot_measured": np.array([[99] * 7, [-4, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0]]),
        },
    }


class ConstantsTest(unittest.TestCase):
    def test_exact_grids_and_scenario(self):
        self.assertEqual(sweep.KP_POS_GRID, [2, 12, 22, 32, 42, 52, 62, 72, 80])
        self.assertEqual(sweep.KD_POS_GRID,
                         [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        self.assertEqual(sweep.SCENARIO.arms, ("right", "left"))
        np.testing.assert_array_equal(
            sweep.SCENARIO.linear_amplitude, [0.18, 0.04, 0.05])
        np.testing.assert_array_equal(
            sweep.SCENARIO.rotational_amplitude, [0.0, 0.0, -0.2])
        self.assertEqual(sweep.SCENARIO.linear_frequency, 0.5)
        self.assertEqual(sweep.SCENARIO.rotational_frequency, 0.5)
        self.assertEqual(sweep.SCENARIO.evaluation_seconds, 10.0)
        self.assertEqual(sweep.SCENARIO.settle_timeout, 20.0)


class MetricTest(unittest.TestCase):
    def test_metrics_use_evaluation_samples_and_keep_arms_separate(self):
        row = sweep.episode_row(_FakeLog(), 12, 0.4)
        right = row["arms"]["right"]
        left = row["arms"]["left"]
        self.assertAlmostEqual(right["position_error_norm_rmse_mm"],
                               np.sqrt((5**2 + 12**2) / 2))
        self.assertEqual(right["position_error_norm_max_mm"], 12.0)
        self.assertEqual(right["peak_measured_joint_speed_rad_s"], 3.0)
        self.assertAlmostEqual(right["ee_linear_speed_norm_rmse_m_s"],
                               np.sqrt((0.5**2 + 1.2**2) / 2))
        self.assertAlmostEqual(left["position_error_norm_rmse_mm"],
                               np.sqrt((10**2 + 0**2) / 2))
        self.assertEqual(left["peak_measured_joint_speed_rad_s"], 4.0)

    def test_missing_or_nonfinite_evaluation_data_marks_episode_invalid(self):
        log = _FakeLog()
        log.evaluation_mask = np.array([False, False, False])
        row = sweep.episode_row(log, 2, 0.2)
        self.assertFalse(row["valid"])
        self.assertIn("missing evaluation samples", row["warning_reasons"])

    def test_nonfinite_evaluation_metrics_are_json_safe_and_invalid(self):
        log = _FakeLog()
        log.arm_data = {
            side: {name: values.astype(float, copy=True)
                   for name, values in fields.items()}
            for side, fields in log.arm_data.items()
        }
        log.arm_data["right"]["e_pos"][1, 0] = np.nan
        log.arm_data["left"]["qdot_measured"][2, 0] = np.inf

        row = sweep.episode_row(log, 2, 0.2)

        self.assertIsNone(
            row["arms"]["right"]["position_error_norm_rmse_mm"])
        self.assertIsNone(
            row["arms"]["left"]["peak_measured_joint_speed_rad_s"])
        self.assertFalse(row["valid"])
        self.assertIn("non-finite data", row["warning_reasons"])
        json.dumps(row, allow_nan=False)

    def test_contact_warning_remains_auditable_without_invalidating_metrics(self):
        log = _FakeLog()
        log.valid = False
        log.warning_reasons = ("contact detected", "negative joint margin")
        row = sweep.episode_row(log, 2, 0.2)
        self.assertTrue(row["valid"])
        self.assertEqual(row["warning_reasons"],
                         ["contact detected", "negative joint margin"])


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_patch = mock.patch.object(sweep, "OUT", Path(self.tmp.name))
        self.out_patch.start()

    def tearDown(self):
        self.out_patch.stop()
        self.tmp.cleanup()

    def test_fingerprint_changes_for_each_required_experiment_input(self):
        baseline = sweep.fingerprint_payload()
        required = ("grid", "scenario", "settling", "evaluation_seconds",
                    "fixed_gains", "metric_schema")
        self.assertEqual(set(baseline), set(required))
        hashes = []
        for key in required:
            changed = json.loads(json.dumps(baseline))
            changed[key] = ["changed"]
            hashes.append(sweep.resume_fingerprint(changed))
        self.assertEqual(len(set(hashes)), len(required))
        self.assertTrue(all(value != sweep.resume_fingerprint(baseline)
                            for value in hashes))

    def test_resume_retries_worker_errors_but_keeps_completed_invalid_rows(self):
        state = sweep.new_state()
        state["results"] = {
            "kp2_kd0.2": {"status": "worker_error", "error": "boom"},
            "kp12_kd0.2": {"status": "completed", "valid": False,
                            "warning_reasons": ["settling timeout"]},
        }
        pending = sweep.pending_jobs(state, [(2, 0.2), (12, 0.2), (22, 0.2)])
        self.assertEqual(pending, [(2, 0.2), (22, 0.2)])

    def test_resume_retries_explicit_pending_rows(self):
        state = sweep.new_state()
        state["results"] = {
            "kp2_kd0.2": {"status": "pending"},
            "kp12_kd0.2": {"status": "completed"},
        }

        pending = sweep.pending_jobs(state, [(2, 0.2), (12, 0.2), (22, 0.2)])

        self.assertEqual(pending, [(2, 0.2), (22, 0.2)])

    def test_mismatched_state_requires_fresh(self):
        state = sweep.new_state()
        state["fingerprint"] = "different"
        sweep.write_state_atomic(state)
        args = argparse.Namespace(fresh=False)
        with self.assertRaisesRegex(RuntimeError, "--fresh"):
            sweep.load_or_init_state(args)

    def test_plots_only_requires_compatible_saved_state(self):
        args = argparse.Namespace(fresh=False, plots_only=True, workers=3)
        with self.assertRaisesRegex(RuntimeError, "plots-only.*saved state"):
            sweep.load_or_init_state(args)

    def test_malformed_saved_state_has_clear_recovery_guidance(self):
        valid = sweep.new_state()
        malformed_states = {
            "missing results mapping": lambda state: state.pop("results"),
            "provenance is not an object": lambda state: state.update(
                provenance=[]),
            "missing timestamp": lambda state: state["provenance"].pop(
                "timestamp"),
            "invalid worker count": lambda state: state["provenance"].update(
                worker_count=0),
            "invalid timestep": lambda state: state["provenance"].update(
                timestep="fast"),
            "invalid dependency versions": lambda state: state[
                "provenance"].update(dependency_versions=[]),
            "invalid git revision": lambda state: state["provenance"].update(
                git_revision=7),
            "invalid execution sessions": lambda state: state[
                "provenance"].update(execution_sessions=[{"timestamp": "now"}]),
            "unknown row status": lambda state: state["results"].update(
                kp2_kd0_2={"status": "mystery"}),
        }
        args = argparse.Namespace(fresh=False, plots_only=False, workers=2)
        for label, corrupt in malformed_states.items():
            with self.subTest(label=label):
                state = json.loads(json.dumps(valid))
                corrupt(state)
                (sweep.OUT / "sweep_state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(
                        RuntimeError, r"malformed.*--fresh"):
                    sweep.load_or_init_state(args)

    def test_metadata_uses_persisted_provenance_without_replacing_origin(self):
        state = sweep.new_state(workers=2, timestamp="original-time")
        sweep.record_execution_session(state, workers=4, timestamp="resume-time")

        sweep.write_metadata(state)
        metadata = json.loads((sweep.OUT / "metadata.json").read_text())

        self.assertEqual(metadata["timestamp"], "original-time")
        self.assertEqual(metadata["worker_count"], 2)
        self.assertEqual(metadata["execution_sessions"], [
            {"timestamp": "original-time", "worker_count": 2},
            {"timestamp": "resume-time", "worker_count": 4},
        ])


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_patch = mock.patch.object(sweep, "OUT", Path(self.tmp.name))
        self.out_patch.start()

    def tearDown(self):
        self.out_patch.stop()
        self.tmp.cleanup()

    def test_summary_has_fixed_schema_and_one_row_for_every_gain_pair(self):
        state = sweep.new_state(workers=2, timestamp="time")
        state["results"]["kp2_kd0.2"] = {
            "config_id": "kp2_kd0.2", "kp_pos": 2.0, "kd_pos": 0.2,
            "status": "worker_error", "error": "RuntimeError: boom",
        }

        sweep.write_summary(state)

        with (sweep.OUT / "summary.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, sweep.SUMMARY_FIELDS)
        self.assertEqual(len(rows), 90)
        self.assertEqual(rows[0]["status"], "worker_error")
        self.assertEqual(rows[1]["status"], "pending")
        for row in rows[:2]:
            self.assertEqual(row["settled"], "")
            self.assertEqual(row["settle_duration_s"], "")
            self.assertEqual(row["warning_reasons"], "")
            self.assertEqual(row["evaluation_sample_count"], "")
            for side in ("right", "left"):
                for metric in sweep.METRIC_SCHEMA:
                    self.assertIn(f"{side}_{metric}", row)


class MatrixAndParallelTest(unittest.TestCase):
    def test_matrix_places_kp_on_columns_and_kd_on_rows(self):
        rows = [
            {"kp_pos": 12, "kd_pos": 0.4, "status": "completed", "valid": True,
             "arms": {"right": {"position_error_norm_rmse_mm": 7}}},
            {"kp_pos": 2, "kd_pos": 0.2, "status": "completed", "valid": False,
             "arms": {"right": {"position_error_norm_rmse_mm": 5}}},
        ]
        values, invalid = sweep.heatmap_matrix(
            rows, "right", "position_error_norm_rmse_mm",
            kp_grid=[2, 12], kd_grid=[0.2, 0.4])
        np.testing.assert_equal(values, [[5, np.nan], [np.nan, 7]])
        np.testing.assert_array_equal(invalid, [[True, True], [True, False]])

    def test_parent_records_each_future_result_and_worker_never_writes_state(self):
        jobs = [(2, 0.2), (12, 0.4)]
        rows = [
            {"kp_pos": kp, "kd_pos": kd, "status": "completed", "valid": True,
             "arms": {}} for kp, kd in jobs
        ]

        class ImmediateFuture:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class FakeExecutor:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.submissions = []
                self.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, job):
                self.submissions.append((function, job))
                return ImmediateFuture(rows[jobs.index(job)])

        state = sweep.new_state()
        writes = []
        with mock.patch.object(sweep, "ProcessPoolExecutor", FakeExecutor), \
             mock.patch.object(sweep, "as_completed", side_effect=lambda fs: fs), \
             mock.patch.object(sweep, "write_state_atomic",
                               side_effect=lambda value: writes.append(value.copy())):
            sweep.run_pending(state, jobs, workers=2)
        self.assertEqual(len(writes), 2)
        self.assertEqual(set(state["results"]), {"kp2_kd0.2", "kp12_kd0.4"})
        executor = FakeExecutor.instances[0]
        self.assertEqual(executor.kwargs["max_workers"], 2)
        self.assertEqual(executor.kwargs["mp_context"].get_start_method(), "spawn")
        self.assertEqual(executor.submissions,
                         [(sweep.run_episode, job) for job in jobs])


class CliTest(unittest.TestCase):
    def test_exact_supported_flags(self):
        parser = sweep.build_parser()
        args = parser.parse_args(["--workers", "3", "--fresh"])
        self.assertEqual(args.workers, 3)
        self.assertTrue(args.fresh)
        self.assertFalse(args.plots_only)
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertEqual(option_strings, {"-h", "--help", "--workers",
                                          "--fresh", "--plots-only"})

    def test_fresh_and_plots_only_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            sweep.build_parser().parse_args(["--fresh", "--plots-only"])

    def test_incompatible_flags_do_not_delete_existing_state(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(sweep, "OUT", Path(directory)):
            state_path = sweep.OUT / "sweep_state.json"
            original = '{"existing": "state"}'
            state_path.write_text(original)

            with self.assertRaises(SystemExit):
                sweep.main(["--fresh", "--plots-only"])

            self.assertEqual(state_path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
