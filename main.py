"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  sampled TOML target source + backend PlantState [trajectory, sim/world]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors; PD + DLS qdot          [controller/reactive_controller]
  -> whole-arm human-distance safety projection  [controller/reactive_controller]
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
from controller.trajectory import (
    IndependentArmTargetSource,
    StaticTargetSource,
)
from plotting.live_cartesian_path import (
    DEFAULT_BUFFER_SAMPLES,
    LiveCartesianPathPublisher,
)
from planning import planner
from runtime_config import CONFIG, print_effective_config
from sim import (
    cylinder_view,
    human_safety_view,
    motion,
    planning_view,
    world,
)
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


def _compose_target_source(static_targets, overrides):
    """Combine independently owned per-arm sources without discarding either."""
    unknown = set(overrides) - set(world.SIDES)
    if unknown:
        raise ValueError(f"unknown target-source arms: {sorted(unknown)}")
    sources = {
        side: StaticTargetSource(static_targets.for_arm(side))
        for side in world.SIDES
    }
    sources.update(overrides)
    return IndependentArmTargetSource(
        right=sources["right"],
        left=sources["left"],
    )


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
    source_overrides = {}
    active_trajectories = [
        (side, CONFIG.target(side).trajectory)
        for side in arms
        if CONFIG.target(side).trajectory is not None
    ]
    look_at_object = None
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
            CONFIG.simulation.look_at_object_motion,
        )
        print_target_trajectory_setup(
            trajectory_side, trajectory, setup
        )
        source_overrides[trajectory_side] = setup.selected_source
        look_at_object = setup.look_at_object
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
    # Collision-aware planning ([planning] in config/control.toml). The plan
    # is built from one plant sample BEFORE the Runner takes over, and is
    # delivered through the ordinary target-source seam, so no controller
    # file is involved. The keep-out router is switched off for planned runs
    # because it would rewrite the planned reference position while passing
    # the planned twist through unchanged (docs/planning.md).
    planning_outcome = None
    runner_keepout = None
    if CONFIG.planning.enabled:
        planned = set(planner.planned_sides(CONFIG.planning))
        clash = planned.intersection(
            side for side, _ in active_trajectories
        )
        if clash:
            raise ValueError(
                f"[planning] and [targets.{sorted(clash)[0]}.trajectory] "
                "both drive that arm; disable one"
            )
        runner_keepout = planner.disabled_keepout()
        plant = world.backend.takeover()
        try:
            planning_outcome = planner.plan_from_state(
                plant,
                world.MOUNT_CALIBRATION,
                cylinder_keepout=runner_keepout,
            )
        finally:
            world.backend.release()
        for side in planned:
            source_overrides[side] = planning_outcome.source.for_arm(side)
        print(planning_view.describe(CONFIG.planning, planning_outcome))

    target_source = _compose_target_source(
        static_targets, source_overrides
    )
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        target_source,
        arms,
        **({} if runner_keepout is None
           else {"cylinder_keepout": runner_keepout}),
    )
    runner.start()
    keepout = runner.cylinder_keepout
    print(cylinder_view.describe(keepout, arms))
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
                if look_at_object is not None:
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
                    viewer.user_scn, keepout, cycle.cylinder_routes)
                if planning_outcome is not None:
                    planning_view.draw(
                        viewer.user_scn,
                        planning_outcome,
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
                    for side, status in cycle.cylinder_routes.items():
                        print(
                            f"        {side:5s} route={status.kind:17s} "
                            f"waypoint {status.waypoint_index + 1}"
                            f"/{status.waypoint_count}  "
                            f"final={status.at_final_waypoint}  "
                            f"target_adjusted={status.target_adjusted}"
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
