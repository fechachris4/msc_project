import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis import orientation_gain_sweep as sweep


class FakeLog:
    arms = ("right", "left")
    settled = True
    settle_duration = 1.25
    warning_reasons = ()
    evaluation_mask = np.array([False, True, True])
    arm_data = {
        "right": {
            "e_rot": np.array([[9, 9, 9], [0.03, 0.04, 0], [0, 0, 0.12]]),
            "e_w": np.array([[9, 9, 9], [0.5, 0, 0], [0, 0, 1.2]]),
            "qdot_measured": np.array([[99] * 7, [1, -2, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 3, 0, 0, 0]]),
        },
        "left": {
            "e_rot": np.array([[9, 9, 9], [0, 0.06, 0.08], [0, 0, 0]]),
            "e_w": np.array([[9, 9, 9], [0, 0.6, 0.8], [0, 0, 0]]),
            "qdot_measured": np.array([[99] * 7, [-4, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0]]),
        },
    }


class ConstantsTest(unittest.TestCase):
    def test_exact_grids_scenario_and_fixed_gains(self):
        self.assertEqual(sweep.KP_ROT_GRID, [2, 12, 22, 32, 42, 52, 62, 72, 80])
        self.assertEqual(sweep.KD_ROT_GRID,
                         [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        self.assertEqual(set(sweep.FIXED_GAIN_NAMES),
                         {"KP_POS", "KD_POS", "K_NULL", "DAMPING"})
        np.testing.assert_array_equal(sweep.SCENARIO.linear_amplitude, [0.18, 0.04, 0.05])
        np.testing.assert_array_equal(sweep.SCENARIO.rotational_amplitude, [0, 0, -0.2])
        self.assertEqual(sweep.SCENARIO.linear_frequency, 0.5)
        self.assertEqual(sweep.SCENARIO.rotational_frequency, 0.5)
        self.assertEqual(sweep.SCENARIO.evaluation_seconds, 4.0)
        self.assertEqual(sweep.SCENARIO.settle_timeout, 20.0)
        self.assertEqual(sweep.OUT, Path("analysis/output/orientation_gain_sweep"))


class MetricTest(unittest.TestCase):
    def test_metrics_use_evaluation_samples_and_keep_arms_separate(self):
        row = sweep.episode_row(FakeLog(), 12, 0.4)
        right = row["arms"]["right"]
        self.assertAlmostEqual(right["orientation_error_norm_rmse_deg"],
                               np.degrees(np.sqrt((0.05**2 + 0.12**2) / 2)))
        self.assertAlmostEqual(right["orientation_error_norm_max_deg"], np.degrees(0.12))
        self.assertEqual(right["peak_measured_joint_speed_rad_s"], 3.0)
        self.assertAlmostEqual(right["ee_angular_speed_norm_rmse_rad_s"],
                               np.sqrt((0.5**2 + 1.2**2) / 2))
        self.assertEqual(row["arms"]["left"]["peak_measured_joint_speed_rad_s"], 4.0)

    def test_missing_or_nonfinite_data_marks_episode_invalid(self):
        log = FakeLog()
        log.evaluation_mask = np.array([False, False, False])
        row = sweep.episode_row(log, 2, 0.2)
        self.assertFalse(row["valid"])
        self.assertIn("missing evaluation samples", row["warning_reasons"])
        log = FakeLog()
        log.arm_data["right"]["e_rot"][1, 0] = np.nan
        row = sweep.episode_row(log, 2, 0.2)
        self.assertFalse(row["valid"])
        self.assertIsNone(row["arms"]["right"]["orientation_error_norm_rmse_deg"])
        json.dumps(row, allow_nan=False)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(sweep, "OUT", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_fingerprint_contains_orientation_inputs(self):
        payload = sweep.fingerprint_payload()
        self.assertEqual(payload["grid"], {"kp_rot": sweep.KP_ROT_GRID,
                                            "kd_rot": sweep.KD_ROT_GRID})
        for key in ("grid", "scenario", "settling", "evaluation_seconds",
                    "fixed_gains", "metric_schema"):
            changed = json.loads(json.dumps(payload))
            changed[key] = ["changed"]
            self.assertNotEqual(sweep.resume_fingerprint(payload),
                                sweep.resume_fingerprint(changed))

    def test_resume_retries_errors_but_keeps_completed_rows(self):
        state = sweep.new_state()
        state["results"] = {
            "kp2_kd0.2": {"status": "worker_error"},
            "kp12_kd0.2": {"status": "completed", "valid": False},
        }
        self.assertEqual(sweep.pending_jobs(state, [(2, 0.2), (12, 0.2), (22, 0.2)]),
                         [(2, 0.2), (22, 0.2)])


class HeatmapTest(unittest.TestCase):
    def test_heatmap_matrix_uses_rotation_coordinates(self):
        rows = [{"status": "completed", "kp_rot": 2, "kd_rot": 0.2, "valid": True,
                 "arms": {"right": {"orientation_error_norm_rmse_deg": 7.0}}},
                {"status": "completed", "kp_rot": 12, "kd_rot": 0.4, "valid": False,
                 "arms": {"right": {"orientation_error_norm_rmse_deg": 9.0}}}]
        values, invalid = sweep.heatmap_matrix(
            rows, "right", "orientation_error_norm_rmse_deg")
        self.assertEqual(values.shape, (10, 9))
        self.assertEqual(values[0, 0], 7.0)
        self.assertTrue(invalid[1, 1])


if __name__ == "__main__":
    unittest.main()
