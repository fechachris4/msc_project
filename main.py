"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  sampled TOML target source + backend PlantState [trajectory, sim/world]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors; PD + DLS qdot          [controller/reactive_controller]
  -> integrate joint-position command (rad)      [controller/position_actuation]
  -> backend.exchange: apply command, mj_step, return next state

usage: mjpython main.py [right|left|both] [--trajectory-plot]

With no positional argument, the arm is read from config/control.toml.
"""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos
from controller.runner import ReactivePositionRunner
from controller.trajectory import StaticDualArmTargetSource
from plotting.live_cartesian_path import (
    DEFAULT_BUFFER_SAMPLES,
    LiveCartesianPathPublisher,
)
from runtime_config import CONFIG, print_effective_config
from sim import cylinder_view, motion, world
from sim.target_trajectory import (
    prepare_target_trajectory,
    print_target_trajectory_setup,
)

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
    active_trajectories = [
        (side, CONFIG.target(side).trajectory)
        for side in arms
        if CONFIG.target(side).trajectory is not None
    ]
    if len(active_trajectories) > 1:
        raise ValueError(
            "simulation currently supports one configured "
            "trajectory arm per run"
        )
    trajectory_buffer_samples = DEFAULT_BUFFER_SAMPLES
    if active_trajectories:
        trajectory_side, trajectory = active_trajectories[0]
        initial_joint_position = (
            CONFIG.simulation.initial_joint_position(trajectory_side)
        )
        setup = prepare_target_trajectory(
            world.backend,
            world.MOUNT_CALIBRATION,
            trajectory_side,
            trajectory,
            initial_joint_position,
            static_targets,
        )
        print_target_trajectory_setup(
            trajectory_side, trajectory, setup
        )
        target_source = setup.source
        show_trajectory = (
            show_trajectory or trajectory.open_live_path_plot
        )
        trace_duration = setup.duration_s
        trajectory_buffer_samples = max(
            DEFAULT_BUFFER_SAMPLES,
            math.ceil(
                trace_duration / world.model.opt.timestep
            )
            + 1,
        )
    else:
        target_source = StaticDualArmTargetSource(static_targets)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        target_source,
        arms,
    )
    runner.start()
    keepout = runner.cylinder_keepout
    print(cylinder_view.describe(keepout, arms))
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
                cycle = runner.cycle()
                desired_pos.show_targets(cycle.resolved_targets)
                # Visualization only: user_scn geometry never contacts the
                # arms. The one shared cylinder stays world-aligned.
                viewer.user_scn.ngeom = 0
                cylinder_view.draw(
                    viewer.user_scn, keepout, cycle.cylinder_routes)
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
                    for side, status in cycle.cylinder_routes.items():
                        print(
                            f"        {side:5s} route={status.kind:17s} "
                            f"waypoint {status.waypoint_index + 1}"
                            f"/{status.waypoint_count}  "
                            f"final={status.at_final_waypoint}  "
                            f"target_adjusted={status.target_adjusted}"
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
