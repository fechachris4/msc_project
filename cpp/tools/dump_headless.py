#!/usr/bin/env python3
"""Python counterpart of srl_headless_trace.

PARITY IS CURRENTLY BROKEN BY DESIGN: the Python side moved cylinder
routing to composition time (see below), while the C++ Runner still
routes per cycle, so the CSV schemas differ (no routed_world_m or
per-cycle route columns here) and ``cpp/tools/verify_parity.sh`` will
refuse on the header mismatch. Restoring parity requires porting the
per-arm composition to the C++ side first; until then this dump stands
alone as the Python reference.

Runs exactly what main.py runs -- configured targets, any per-arm
trajectory or planned path, cylinder-keepout routing, and the whole-arm
human-safety filter all enabled -- headlessly for N cycles and dumps the
quantities that path produces.

Composition (``arm_flow.build_flows``) happens BEFORE timing now: a
keep-out routed reach is built as an ordinary timed waypoint trajectory
at composition time, not rewritten per cycle, so the resolved target
column IS the tracked target -- there is no separate "routed" target to
record. What changed per arm at composition time (a plain static hold,
a configured trajectory, a planned path, or a routed reach) is instead
recorded once per row from the composed ``DualArmFlow``, since it does
not vary cycle to cycle.

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

import arm_flow  # noqa: E402
from controller import desired_pos  # noqa: E402
from controller.runner import ReactivePositionRunner  # noqa: E402
from sim import motion, world  # noqa: E402

ARMS = ("right", "left")


def g(value):
    return format(float(value), ".17g")


def names():
    columns = ["cycle", "arm", "sample_time_s"]
    for prefix, count in (
        ("q", 7), ("qdot_measured", 7), ("qdot_raw", 7),
        ("qdot_safety_filtered", 7), ("ctrl_after", 7),
        ("e_pos", 3), ("e_rot", 3),
        ("target_world_m", 3),
    ):
        columns.extend(f"{prefix}_{index}" for index in range(count))
    columns.extend([
        "min_clearance_m", "active_count", "human_adjusted", "limit_adjusted",
        "stopped", "reason",
        # Composition-time facts (arm_flow.DualArmFlow), constant per arm
        # across the whole run -- routing now happens before timing, so
        # "target_world_m" above already IS the routed/tracked target.
        "flow_kind", "route_kind", "waypoint_count",
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

    flow = arm_flow.build_flows(
        world.backend, world.MOUNT_CALIBRATION, static_targets, ARMS
    )

    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        flow.source,
        ARMS,
    )
    runner.start()

    rows = []
    try:
        for cycle in range(args.steps):
            # Mirrors main.py: a look-at-object trajectory must be moved
            # to this visible cycle's time before it is sampled.
            for look_at_object in flow.look_at_objects:
                look_at_object.apply(runner.target_elapsed_time_s)
            result = runner.cycle()
            for side in ARMS:
                one = flow.for_arm(side)
                trace = result.traces[side]
                safety = result.human_safety_statuses[side]
                route = one.route
                row = [str(cycle), side, g(result.input_state.sample_time_s)]
                for value in (
                    trace.q, trace.qdot_measured, trace.qdot_raw,
                    trace.qdot_safety_filtered, trace.ctrl_after,
                    trace.e_pos, trace.e_rot,
                    result.resolved_targets.for_arm(side).pose_world.position_m,
                ):
                    row.extend(g(item) for item in np.asarray(value).reshape(-1))
                row.append(g(safety.minimum_clearance_m))
                row.append(str(safety.active_constraint_count))
                row.append(str(bool(safety.human_adjusted)))
                row.append(str(bool(safety.limit_adjusted)))
                row.append(str(bool(safety.stopped)))
                row.append(safety.reason)
                row.append(one.kind)
                row.append(route.kind if route is not None else "none")
                row.append(
                    str(len(route.waypoints_world_m))
                    if route is not None else "0"
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
