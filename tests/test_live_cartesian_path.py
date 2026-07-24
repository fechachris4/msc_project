from queue import Queue
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from plotting.live_cartesian_path import (
    CartesianPathBuffer,
    LiveCartesianPathPublisher,
    offer_latest_nonblocking,
)


class _ByArm:
    def __init__(self, right, left):
        self.right = right
        self.left = left

    def for_arm(self, side):
        return getattr(self, side)


def _cycle(offset):
    sampled = DualArmFramedTargets(
        FramedTarget(
            TargetFrame.WORLD,
            Pose(np.array([0.4, -0.2, 1.1]), np.eye(3)),
            Twist.zero(),
        ),
        FramedTarget(
            TargetFrame.TORSO,
            Pose(np.array([0.3, 0.2, 0.9]), np.eye(3)),
            Twist.zero(),
        ),
    )
    desired = _ByArm(
        SimpleNamespace(
            pose_world=Pose(
                np.array([0.4 + offset, -0.2, 1.1]), np.eye(3)
            )
        ),
        SimpleNamespace(
            pose_world=Pose(
                np.array([0.3, 0.2 + offset, 1.0]), np.eye(3)
            )
        ),
    )
    measured = _ByArm(
        SimpleNamespace(
            ee_pose_world=Pose(
                np.array([0.39 + offset, -0.2, 1.1]), np.eye(3)
            )
        ),
        SimpleNamespace(
            ee_pose_world=Pose(
                np.array([0.3, 0.19 + offset, 1.0]), np.eye(3)
            )
        ),
    )
    return SimpleNamespace(
        sampled_targets=sampled,
        resolved_targets=desired,
        controller_states=measured,
    )


class LiveCartesianPathDataTest(unittest.TestCase):
    def test_same_cycle_buffer_is_bounded_and_keeps_source_frames(self):
        buffer = CartesianPathBuffer(
            ("right", "left"), max_samples=2
        )
        for offset in (0.0, 0.1, 0.2):
            buffer.append(_cycle(offset))
        snapshot = buffer.snapshot()

        self.assertEqual(buffer.sample_count, 2)
        self.assertEqual(
            snapshot.source_frames,
            {"right": "world", "left": "torso"},
        )
        np.testing.assert_allclose(
            snapshot.desired_world_m["right"][:, 0],
            [0.5, 0.6],
        )
        np.testing.assert_allclose(
            snapshot.measured_world_m["right"][:, 0],
            [0.49, 0.59],
        )

    def test_full_channel_is_replaced_without_a_blocking_put(self):
        channel = Queue(maxsize=1)
        channel.put_nowait("stale")

        self.assertTrue(
            offer_latest_nonblocking(channel, "latest")
        )
        self.assertEqual(channel.get_nowait(), "latest")

    def test_noninteractive_backend_fails_before_starting_process(self):
        publisher = LiveCartesianPathPublisher(("right",))
        with mock.patch(
            "matplotlib.get_backend", return_value="Agg"
        ), self.assertRaisesRegex(RuntimeError, "not interactive"):
            publisher.start()
        publisher.close()


if __name__ == "__main__":
    unittest.main()
