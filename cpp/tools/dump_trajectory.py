#!/usr/bin/env python3
"""Python counterpart of srl_trajectory_dump, for trajectory parity checking.

    .venv/bin/python cpp/tools/dump_trajectory.py CONFIG.toml [SAMPLES]

Emits the same line format so the two outputs can be diffed field by field.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from controller.state import Pose  # noqa: E402
from controller.trajectory_config import materialize_trajectory  # noqa: E402
from runtime_config import load_config  # noqa: E402


def g(value):
    return format(float(value), ".17g")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: dump_trajectory.py CONFIG.toml [SAMPLES]")
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    config = load_config(sys.argv[1])
    trajectory = config.left_target.trajectory
    if trajectory is None:
        raise SystemExit("config has no left trajectory")

    start_pose = Pose(np.array([0.45, 0.30, 1.20]), np.eye(3))
    materialized = materialize_trajectory(trajectory, start_pose)

    print(f"duration,{g(materialized.duration_s)}")
    for boundary in materialized.boundary_times_s:
        print(f"boundary,{g(boundary)}")
    bounds = materialized.rate_bounds
    print(
        "bounds,"
        f"{g(bounds.max_linear_speed_m_s)},"
        f"{g(bounds.max_linear_acceleration_m_s2)},"
        f"{g(bounds.max_angular_speed_rad_s)},"
        f"{g(bounds.max_angular_acceleration_rad_s2)}"
    )

    horizon = materialized.duration_s * 1.05
    source = materialized.source
    for index in range(samples + 1):
        t = horizon * index / samples
        sample = source.sample_kinematics(t)
        pose = sample.target.pose
        twist = sample.target.twist
        print(f"pos,{g(t)},{g(pose.position_m[0])},{g(pose.position_m[1])},"
              f"{g(pose.position_m[2])}")
        for row in range(3):
            print(f"rot,{g(t)},{g(pose.rotation[row, 0])},"
                  f"{g(pose.rotation[row, 1])},{g(pose.rotation[row, 2])}")
        print(f"vel,{g(t)},{g(twist.linear_m_s[0])},{g(twist.linear_m_s[1])},"
              f"{g(twist.linear_m_s[2])}")
        print(f"omg,{g(t)},{g(twist.angular_rad_s[0])},"
              f"{g(twist.angular_rad_s[1])},{g(twist.angular_rad_s[2])}")
        acc = sample.linear_acceleration_m_s2
        print(f"acc,{g(t)},{g(acc[0])},{g(acc[1])},{g(acc[2])}")
        aac = sample.angular_acceleration_rad_s2
        print(f"aac,{g(t)},{g(aac[0])},{g(aac[1])},{g(aac[2])}")


if __name__ == "__main__":
    main()
