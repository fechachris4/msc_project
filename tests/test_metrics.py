from contextlib import redirect_stdout
import io
import unittest

import numpy as np


class MetricsTest(unittest.TestCase):
    def test_stats_and_window_are_static_base_safe(self):
        from analysis.metrics import stats, windowed_stats

        legacy = stats({
            "base_disp": np.zeros((2, 3)),
            "right_e": np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]]),
            "left_e": np.zeros((2, 3)),
        })

        self.assertEqual(set(legacy), {"peak_base", "right", "left"})
        self.assertIsNone(legacy["right"]["rejection"])
        self.assertIsNone(legacy["left"]["rejection"])
        self.assertEqual(windowed_stats(
            np.array([1.0, 2.0]), np.zeros(2)),
            (np.sqrt(2.5), 2.0, None),
        )

    def test_print_stats_alignment_and_none(self):
        from analysis.metrics import print_stats

        values = {
            "peak_base": 2.0,
            "right": {
                "rms": np.zeros(3), "peak": np.zeros(3),
                "peak_norm": 1.0, "rejection": 50.0,
            },
            "left": {
                "rms": np.zeros(3), "peak": np.zeros(3),
                "peak_norm": 0.0, "rejection": None,
            },
        }
        stream = io.StringIO()

        with redirect_stdout(stream):
            print_stats(values)

        lines = stream.getvalue().splitlines()
        self.assertTrue(lines[1].endswith(f"{50.0:11.1f}%  [mm]"))
        self.assertTrue(lines[2].endswith(f"{'n/a':>12s}  [mm]"))


if __name__ == "__main__":
    unittest.main()
