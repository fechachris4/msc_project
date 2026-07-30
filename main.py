"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  per-arm composed reference source + backend PlantState   [arm_flow]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors; PD + DLS qdot          [controller/reactive_controller]
  -> whole-arm human-distance safety projection  [controller/reactive_controller]
  -> integrate joint-position command (rad)      [controller/position_actuation]
  -> backend.exchange: apply command, mj_step, return next state

Each arm's reference is composed independently in arm_flow.py (static
target, configured trajectory, planned path, or keep-out routed reach);
the Runner is the thin two-arm layer that samples the composed source
once per cycle and never rewrites it.

usage: mjpython main.py [right|left|both] [--trajectory-plot]

With no positional argument, the arm is read from config/control.toml.
"""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import arm_flow
from controller import desired_pos
from controller.runner import ReactivePositionRunner
from plotting.live_cartesian_path import (
    DEFAULT_BUFFER_SAMPLES,
    LiveCartesianPathPublisher,
)
from runtime_config import CONFIG, print_effective_config
from sim import (
    cylinder_view,
    human_safety_view,
    motion,
    planning_view,
    world,
)
from sim.target_trajectory import print_target_trajectory_setup

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

def _parse_args(argv, default_choice="both"):
    values = list(argv)
    usage = "usage: mjpython main.py [right|left|both]"
    if len(values) > 1:
        raise SystemExit(usage)
    choice = values[0] if values else default_choice
    if choice not in ("right", "left", "both"):
        raise SystemExit(usage)
    return world.SIDES if choice == "both" else (choice,)


def _parse_options(argv, default_choice="both"):
    values = list(argv)
    flag = "--trajectory-plot"
    if values.count(flag) > 1:
        raise SystemExit(
            "usage: mjpython main.py [right|left|both] "
            "[--trajectory-plot]"
        )
    show_trajectory = flag in values
    positional = [value for value in values if value != flag]
    return _parse_args(positional, default_choice), show_trajectory


def main(argv=None):
    import sys

    arms, show_trajectory = _parse_options(
        sys.argv[1:] if argv is None else argv,
        default_choice=CONFIG.run.arm,
    )
    print_effective_config(CONFIG)
    static_targets = desired_pos.configured_targets()
    world.backend.configure_torso_driver(
        motion.torso_pose_at, motion.torso_twist_at)

    # Each arm's reference is composed independently from its own config
    # and one shared plant snapshot; planning or routing one arm cannot
    # replace or freeze the other arm's configured motion.
    flow = arm_flow.build_flows(
        world.backend,
        world.MOUNT_CALIBRATION,
        static_targets,
        arms,
    )
    for side in world.SIDES:
        one = flow.for_arm(side)
        if one.kind == "trajectory":
            trajectory = CONFIG.target(side).trajectory
            print_target_trajectory_setup(
                side, trajectory, one.trajectory_setup
            )
            show_trajectory = (
                show_trajectory or trajectory.open_live_path_plot
            )
    if flow.plans:
        print(planning_view.describe(CONFIG.planning, flow.plans))

    trajectory_buffer_samples = DEFAULT_BUFFER_SAMPLES
    if flow.duration_s > 0.0:
        trajectory_buffer_samples = max(
            DEFAULT_BUFFER_SAMPLES,
            math.ceil(flow.duration_s / world.model.opt.timestep) + 1,
        )

    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        flow.source,
        arms,
    )
    runner.start()
    keepout = arm_flow.keepout_from_config(CONFIG.cylinder_keepout)
    print(cylinder_view.describe(keepout, arms))
    for side, route in flow.routes.items():
        print(
            f"  {side} reach routed {route.kind} around the keep-out: "
            f"{len(route.waypoints_world_m)} waypoints, "
            f"{flow.for_arm(side).duration_s:.2f} s timed path"
        )
    print(human_safety_view.describe(CONFIG.human_safety))
    path_publisher = None
    path_publication_enabled = False
    if show_trajectory:
        path_publisher = LiveCartesianPathPublisher(
            arms, max_samples=trajectory_buffer_samples
        )
        try:
            path_publisher.start()
            path_publication_enabled = True
        except Exception as error:
            print(
                "Live trajectory plot unavailable; continuing control "
                f"without it: {error}"
            )

    step = 0
    try:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                for look_at_object in flow.look_at_objects:
                    # The object is moved and then sampled at this same
                    # visible-cycle time by the structured target source.
                    look_at_object.apply(
                        runner.target_elapsed_time_s
                    )
                cycle = runner.cycle()
                desired_pos.show_targets(cycle.resolved_targets)
                # Visualization only: user_scn geometry never contacts the
                # arms. The one shared cylinder stays world-aligned.
                viewer.user_scn.ngeom = 0
                cylinder_view.draw(
                    viewer.user_scn, keepout, flow.routes or None)
                if flow.plans:
                    planning_view.draw(
                        viewer.user_scn,
                        flow.plans,
                        cycle.input_state.torso_pose_world,
                    )
                human_safety_view.draw(
                    viewer.user_scn,
                    cycle.input_state,
                    cycle.controller_states,
                    cycle.human_safety_states,
                    CONFIG.human_safety,
                    arms,
                )
                if path_publication_enabled:
                    try:
                        path_publisher.append(cycle)
                    except Exception as error:
                        path_publication_enabled = False
                        print(
                            "Live trajectory publication stopped; "
                            f"control continues unchanged: {error}"
                        )

                if step % PRINT_EVERY == 0:
                    for side, trace in cycle.traces.items():
                        e_pos = trace.e_pos
                        state = cycle.input_state
                        sample_time = state.sample_time_s
                        e_mm = e_pos * 1000.0
                        sigma = np.linalg.svd(
                            trace.J, compute_uv=False)
                        print(
                            f"t={sample_time:6.2f}s  {side:5s} "
                            f"|e|={np.linalg.norm(e_mm):.1f} mm  "
                            f"e_pos=[{e_mm[0]: 7.1f} {e_mm[1]: 7.1f} "
                            f"{e_mm[2]: 7.1f}]  "
                            f"sigma=[{' '.join(f'{s:.3f}' for s in sigma)}]"
                        )
                    for side, status in (
                        cycle.human_safety_statuses.items()
                    ):
                        clearance_mm = (
                            status.minimum_clearance_m * 1000.0
                        )
                        label = (
                            "SAFETY STOP"
                            if status.stopped
                            else "human safety"
                        )
                        print(
                            f"        {side:5s} {label}: "
                            f"clearance={clearance_mm:.1f} mm  "
                            f"active={status.active_constraint_count}  "
                            f"adjusted={status.human_adjusted}  "
                            f"reason={status.reason}"
                        )
                    diagnostic = cylinder_view.format_link_intersections(
                        cylinder_view.link_intersections(
                            world.model, world.data, keepout)
                    )
                    if diagnostic is not None:
                        print(f"        {diagnostic}")
                step += 1

                viewer.sync()

                time_until_next_step = (
                    world.model.opt.timestep
                    - (time.perf_counter() - step_start)
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)
        if path_publisher is not None:
            path_publisher.close()


if __name__ == "__main__":
    main()
