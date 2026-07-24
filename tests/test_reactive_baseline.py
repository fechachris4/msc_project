from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from analysis import reactive_baseline
from sim import motion


class ReactiveBaselineArtifactTest(unittest.TestCase):
    def test_scenario_covers_two_periods_and_current_motion(self):
        scenario = reactive_baseline.SCENARIO
        np.testing.assert_array_equal(
            scenario.linear_amplitude, motion.LINEAR_AMPLITUDE)
        np.testing.assert_array_equal(
            scenario.rotational_amplitude, motion.ROTATIONAL_AMPLITUDE)
        longest_period = max(
            1.0 / scenario.linear_frequency,
            1.0 / scenario.rotational_frequency,
        )
        self.assertGreaterEqual(
            scenario.evaluation_seconds, 2.0 * longest_period)

    def test_figure_contains_trace_and_summary_inputs(self):
        log = SimpleNamespace(
            evaluation_mask=np.array([True, True, True]),
            eval_time=np.array([0.0, 0.002, 0.004]),
            base_displacement=np.array([
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
            ]),
            arm_data={
                "right": {"e_pos": np.full((3, 3), 0.001)},
                "left": {"e_pos": np.full((3, 3), 0.002)},
            },
        )
        run_metrics = {
            "arms": {
                "right": {
                    "position_error_norm_mean_m": 0.001,
                    "position_error_norm_rmse_m": 0.002,
                    "position_error_norm_peak_m": 0.003,
                },
                "left": {
                    "position_error_norm_mean_m": 0.004,
                    "position_error_norm_rmse_m": 0.005,
                    "position_error_norm_peak_m": 0.006,
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = reactive_baseline.make_figure(
                log, run_metrics, Path(temporary) / "baseline.png")
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
