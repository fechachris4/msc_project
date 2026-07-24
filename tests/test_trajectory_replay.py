"""Cross-language-ready replay gate for Cartesian waypoint sampling."""

import json
from pathlib import Path
import unittest

import numpy as np

from controller.state import Pose, TargetFrame
from controller.trajectory import (
    CartesianWaypoint,
    CartesianWaypointTrajectory,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "golden"
    / "cartesian_waypoint_replay.json"
)


class CartesianWaypointReplayTest(unittest.TestCase):
    def test_fixture_samples_match_portable_numeric_contract(self):
        data = json.loads(FIXTURE.read_text())
        self.assertEqual(data["schema_version"], 2)
        self.assertIn("minimum integrated squared", data["numeric_convention"])
        trajectory_data = data["trajectory"]
        trajectory = CartesianWaypointTrajectory(
            TargetFrame(trajectory_data["reference_frame"]),
            tuple(
                CartesianWaypoint(
                    item["time_s"],
                    Pose(
                        item["position_m"],
                        np.asarray(
                            item["rotation_row_major"], dtype=float
                        ).reshape(3, 3),
                    ),
                )
                for item in trajectory_data["waypoints"]
            ),
        )
        rtol = data["comparison"]["relative_tolerance"]
        atol = data["comparison"]["absolute_tolerance"]

        for expected in data["samples"]:
            with self.subTest(
                elapsed_time_s=expected["elapsed_time_s"]
            ):
                kinematics = trajectory.sample_kinematics(
                    expected["elapsed_time_s"]
                )
                actual = kinematics.target
                self.assertEqual(
                    actual.reference_frame,
                    TargetFrame(trajectory_data["reference_frame"]),
                )
                fields = (
                    (actual.pose.position_m, expected["position_m"]),
                    (
                        actual.pose.rotation.reshape(-1),
                        expected["rotation_row_major"],
                    ),
                    (
                        actual.twist.linear_m_s,
                        expected["linear_velocity_m_s"],
                    ),
                    (
                        actual.twist.angular_rad_s,
                        expected["angular_velocity_rad_s"],
                    ),
                    (
                        kinematics.linear_acceleration_m_s2,
                        expected["linear_acceleration_m_s2"],
                    ),
                    (
                        kinematics.angular_acceleration_rad_s2,
                        expected["angular_acceleration_rad_s2"],
                    ),
                )
                for value, reference in fields:
                    np.testing.assert_allclose(
                        value, reference, rtol=rtol, atol=atol
                    )


if __name__ == "__main__":
    unittest.main()
