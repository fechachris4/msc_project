from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.cartesian_path import make_cartesian_path_figure


class CartesianPathArtifactTest(unittest.TestCase):
    def test_static_and_moving_desired_paths_generate_readable_artifact(self):
        sample_count = 12
        static_desired = np.tile([0.4, -0.2, 1.1], (sample_count, 1))
        parameter = np.linspace(0.0, 1.0, sample_count)
        moving_desired = np.column_stack([
            0.3 + 0.1 * parameter,
            0.2 + 0.04 * np.sin(np.pi * parameter),
            1.0 + 0.05 * parameter,
        ])
        log = SimpleNamespace(
            arms=("right", "left"),
            target_reference_frames={
                "right": "world",
                "left": "torso",
            },
            arm_data={
                "right": {
                    "target_position_world_m": static_desired,
                    "ee_position_world_m": (
                        static_desired
                        + np.column_stack([
                            0.002 * parameter,
                            np.zeros(sample_count),
                            -0.001 * parameter,
                        ])
                    ),
                },
                "left": {
                    "target_position_world_m": moving_desired,
                    "ee_position_world_m": (
                        moving_desired
                        + np.column_stack([
                            np.zeros(sample_count),
                            -0.002 * parameter,
                            0.001 * parameter,
                        ])
                    ),
                },
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = make_cartesian_path_figure(
                log, Path(temporary) / "cartesian_path.png"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
