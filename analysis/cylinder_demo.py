"""Cylinder keep-out demonstration: start and target on opposite sides.

Places one world-vertical keep-out cylinder directly between the right arm's
measured start pose and a target mirrored to the far side of it, so the direct
path is blocked and the router must go around (or over). This diagnostic uses
local geometry and never changes the committed central-person configuration.

    .venv/bin/mjpython -m analysis.cylinder_demo            # viewer (macOS)
    .venv/bin/python -m analysis.cylinder_demo --headless   # no window

Everything is SI (metres, radians); millimetres appear only in the printout.

SCOPE: end-effector waypoint routing only, exactly as on hardware. This is
NOT whole-arm or per-link collision avoidance. Link intersections with the
drawn volume are reported as diagnostics and never change the route.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller.cylinder_router import CylinderKeepout
from controller.runner import ReactivePositionRunner
from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from runtime_config import CONFIG
from sim import cylinder_view, world


SIDE = "right"
# The demo arm starts from a bent, non-singular posture. With MuJoCo's default
# all-zero joints the Gen3 stands fully extended (1.19 m from base_link along
# base +z), which is a workspace-boundary singularity: the arm then cannot
# track lateral waypoints, and the demonstration would show controller
# residuals rather than routing. This is the posture config/control.toml
# already ships for the left arm.
START_POSTURE_RAD = CONFIG.simulation.initial_joint_position("left")

# Cylinder geometry for the demonstration, metres in the world frame.
# The route direction is world +y, verified reachable at 0.00 mm residual from
# the posture above; world +-x is near the reach limit and would confound the
# demonstration with tracking error.
ROUTE_AXIS = np.array([0.0, 1.0, 0.0])
STANDOFF_M = 0.15   # cylinder centre this far along ROUTE_AXIS from the EE
RADIUS_M = 0.08
CLEARANCE_M = 0.02
HALF_HEIGHT_M = 0.50
WAYPOINT_TOLERANCE_M = 0.02
PRINT_EVERY = 250


def set_start_posture():
    """Seed the demo arm in a bent, non-singular configuration."""
    world.backend.release()
    world.backend.reset()
    world.backend.data.qpos[world.backend.qpos_adrs[SIDE]] = np.asarray(
        START_POSTURE_RAD, dtype=float)
    world.backend.data.qvel[:] = 0.0
    mujoco.mj_forward(world.backend.model, world.backend.data)


def build_scenario():
    """Return (keepout, framed targets, start/target in world frame).

    The target is the measured start position reflected through the cylinder
    axis, so start and target sit on exactly opposite sides of it and the
    straight line between them passes through the keep-out.
    """
    from controller import frames

    set_start_posture()
    plant = world.read_state(Twist.zero())
    states = frames.controller_states(plant, world.MOUNT_CALIBRATION)
    ee = states.for_arm(SIDE).ee_pose_world

    start_world = np.asarray(ee.position_m, dtype=float)
    rotation_world = np.asarray(ee.rotation, dtype=float)

    axis = ROUTE_AXIS / float(np.linalg.norm(ROUTE_AXIS))
    centre = start_world + STANDOFF_M * axis
    # Mirror the start through the centre: opposite sides, equal standoff.
    target_world = centre + STANDOFF_M * axis

    keepout = CylinderKeepout(
        enabled=True,
        center_xy_m=(float(centre[0]), float(centre[1])),
        radius_m=RADIUS_M,
        z_min_m=float(start_world[2]) - HALF_HEIGHT_M,
        z_max_m=float(start_world[2]) + HALF_HEIGHT_M,
        clearance_m=CLEARANCE_M,
        waypoint_tolerance_m=WAYPOINT_TOLERANCE_M,
    )

    demo_target = FramedTarget(
        TargetFrame.WORLD,
        Pose(target_world, rotation_world),
        Twist.zero(),
    )
    held = FramedTarget(
        TargetFrame.WORLD,
        Pose(start_world, rotation_world),
        Twist.zero(),
    )
    targets = (
        DualArmFramedTargets(right=demo_target, left=held)
        if SIDE == "right"
        else DualArmFramedTargets(right=held, left=demo_target)
    )
    return keepout, targets, start_world, target_world


def _print_header(keepout, start_world, target_world):
    print(cylinder_view.describe(keepout, (SIDE,)))
    separation = float(np.linalg.norm(target_world[:2] - start_world[:2]))
    print(
        f"  start  (world) = [{start_world[0]: .3f} {start_world[1]: .3f} "
        f"{start_world[2]: .3f}] m"
    )
    print(
        f"  target (world) = [{target_world[0]: .3f} {target_world[1]: .3f} "
        f"{target_world[2]: .3f}] m"
    )
    print(
        f"  centre-to-centre separation = {separation:.3f} m across an "
        f"inflated radius of {keepout.obstacle_radius_m:.3f} m"
    )


def _print_status(step, cycle, keepout):
    status = cycle.cylinder_routes.get(SIDE)
    if status is None:
        return
    error_mm = (
        float(np.linalg.norm(cycle.traces[SIDE].e_pos)) * 1000.0
        if SIDE in cycle.traces
        else 0.0
    )
    print(
        f"t={cycle.input_state.sample_time_s:6.2f}s  route={status.kind:17s} "
        f"waypoint {status.waypoint_index + 1}/{status.waypoint_count}  "
        f"final={str(status.at_final_waypoint):5s}  "
        f"adjusted={str(status.target_adjusted):5s}  |e|={error_mm:7.1f} mm"
    )
    message = cylinder_view.format_link_intersections(
        cylinder_view.link_intersections(
            world.model, world.data, keepout)
    )
    if message is not None:
        print(f"    {message}")


def run(headless=False, steps=4000):
    keepout, targets, start_world, target_world = build_scenario()
    _print_header(keepout, start_world, target_world)

    world.backend.configure_torso_driver(None, None)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        targets,
        arms=(SIDE,),
        cylinder_keepout=keepout,
    )
    runner.start()

    kinds = set()
    try:
        if headless:
            for step in range(steps):
                cycle = runner.cycle()
                status = cycle.cylinder_routes.get(SIDE)
                if status is not None:
                    kinds.add(status.kind)
                if step % PRINT_EVERY == 0:
                    _print_status(step, cycle, keepout)
            return kinds

        with mujoco.viewer.launch_passive(
            world.model, world.data
        ) as viewer:
            step = 0
            while viewer.is_running():
                step_start = time.perf_counter()
                cycle = runner.cycle()
                viewer.user_scn.ngeom = 0
                cylinder_view.draw(
                    viewer.user_scn, keepout, cycle.cylinder_routes)
                status = cycle.cylinder_routes.get(SIDE)
                if status is not None:
                    kinds.add(status.kind)
                if step % PRINT_EVERY == 0:
                    _print_status(step, cycle, keepout)
                step += 1
                viewer.sync()
                remaining = (
                    world.model.opt.timestep
                    - (time.perf_counter() - step_start)
                )
                if remaining > 0:
                    time.sleep(remaining)
        return kinds
    finally:
        runner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true",
        help="run without opening the MuJoCo viewer")
    parser.add_argument(
        "--steps", type=int, default=4000,
        help="headless step count (default 4000 = 8 s at the 2 ms timestep)")
    args = parser.parse_args(argv)
    kinds = run(headless=args.headless, steps=args.steps)
    print(f"route kinds used: {sorted(kinds)}")


if __name__ == "__main__":
    main()
