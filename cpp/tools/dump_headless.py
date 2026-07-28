#!/usr/bin/env python3
"""Python counterpart of srl_headless_trace.

Runs exactly what main.py runs -- configured targets, the configured
trajectory arm, cylinder routing and the whole-arm human-safety filter all
enabled -- headlessly for N cycles and dumps the quantities that path
produces.

    .venv/bin/python cpp/tools/dump_headless.py --out PATH [--steps N]

This is the parity harness for the code the golden trace does NOT cover:
tests/golden_trace.py deliberately disables human safety.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from controller import desired_pos  # noqa: E402
from controller.runner import ReactivePositionRunner  # noqa: E402
from controller.trajectory import StaticDualArmTargetSource  # noqa: E402
from runtime_config import CONFIG  # noqa: E402
from sim import motion, world  # noqa: E402
from sim.target_trajectory import prepare_target_trajectory  # noqa: E402

ARMS = ("right", "left")


def g(value):
    return format(float(value), ".17g")


def names():
    columns = ["cycle", "arm", "sample_time_s"]
    for prefix, count in (
        ("q", 7), ("qdot_measured", 7), ("qdot_raw", 7),
        ("qdot_safety_filtered", 7), ("ctrl_after", 7),
        ("e_pos", 3), ("e_rot", 3),
        ("target_world_m", 3), ("routed_world_m", 3),
    ):
        columns.extend(f"{prefix}_{index}" for index in range(count))
    columns.extend([
        "min_clearance_m", "active_count", "human_adjusted", "limit_adjusted",
        "stopped", "reason", "route_kind", "waypoint_count", "target_adjusted",
    ])
    return columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/py_headless.csv")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    static_targets = desired_pos.configured_targets()
    world.backend.configure_torso_driver(
        motion.torso_pose_at, motion.torso_twist_at)

    active = [
        (side, CONFIG.target(side).trajectory)
        for side in ARMS
        if CONFIG.target(side).trajectory is not None
    ]
    if len(active) > 1:
        raise SystemExit("one configured trajectory arm per run")
    if active:
        side, trajectory = active[0]
        setup = prepare_target_trajectory(
            world.backend,
            world.MOUNT_CALIBRATION,
            side,
            trajectory,
            CONFIG.simulation.initial_joint_position(side),
            static_targets,
        )
        target_source = setup.source
    else:
        target_source = StaticDualArmTargetSource(static_targets)

    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        target_source,
        ARMS,
    )
    runner.start()

    rows = []
    try:
        for cycle in range(args.steps):
            result = runner.cycle()
            for side in ARMS:
                trace = result.traces[side]
                safety = result.human_safety_statuses[side]
                route = result.cylinder_routes.get(side)
                row = [str(cycle), side, g(result.input_state.sample_time_s)]
                for value in (
                    trace.q, trace.qdot_measured, trace.qdot_raw,
                    trace.qdot_safety_filtered, trace.ctrl_after,
                    trace.e_pos, trace.e_rot,
                    result.resolved_targets.for_arm(side).pose_world.position_m,
                    result.routed_targets.for_arm(side).pose_world.position_m,
                ):
                    row.extend(g(item) for item in np.asarray(value).reshape(-1))
                row.append(g(safety.minimum_clearance_m))
                row.append(str(safety.active_constraint_count))
                row.append(str(bool(safety.human_adjusted)))
                row.append(str(bool(safety.limit_adjusted)))
                row.append(str(bool(safety.stopped)))
                row.append(safety.reason)
                row.append(route.kind if route is not None else "none")
                row.append(str(route.waypoint_count if route is not None else 0))
                row.append(
                    str(bool(route.target_adjusted))
                    if route is not None else "False"
                )
                rows.append(row)
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)

    with Path(args.out).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names())
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
