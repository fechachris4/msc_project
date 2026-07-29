"""Simulation initialization for configured target trajectories."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from controller import frames
from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import (
    IndependentArmTargetSource,
    StaticTargetSource,
)
from controller.trajectory_config import materialize_trajectory
from sim.look_at_object import SimulatedMocapPointSource


@dataclass(frozen=True, slots=True)
class TargetTrajectorySetup:
    source: IndependentArmTargetSource
    selected_source: object
    start_pose_reference: Pose
    end_pose_reference: Pose
    start_pose_world: Pose
    end_pose_world: Pose
    duration_s: float
    boundary_times_s: tuple[float, ...]
    rate_bounds: object
    initial_singular_values: np.ndarray
    initial_joint_margin_rad: float
    look_at_object: SimulatedMocapPointSource | None


def _limited_joint_margin(backend, side, joint_position_rad):
    lower, upper, limited = backend.jnt_range(side)
    margins = np.minimum(
        joint_position_rad - lower,
        upper - joint_position_rad,
    )[limited]
    return float(np.min(margins)) if margins.size else math.inf


def _set_initial_joint_posture(
    backend, side, initial_joint_position_rad
):
    backend.release()
    backend.reset()
    if initial_joint_position_rad is None:
        mujoco.mj_forward(backend.model, backend.data)
        return
    joint_position = np.asarray(
        initial_joint_position_rad, dtype=float
    )
    backend.data.qpos[backend.qpos_adrs[side]] = joint_position
    backend.data.qvel[:] = 0.0
    mujoco.mj_forward(backend.model, backend.data)


def _configured_start_pose(
    plant,
    side,
    calibration,
    reference_frame,
    trajectory,
    static_targets,
    measured_world,
):
    if trajectory.start == "measured":
        world_pose = measured_world
    elif trajectory.start == "configured_target":
        world_pose = frames.resolve_target_world(
            plant,
            side,
            calibration,
            static_targets.for_arm(side),
        ).pose_world
    else:
        raise ValueError(
            f"unsupported trajectory start: {trajectory.start!r}"
        )
    return frames.express_pose_in_target_frame(
        plant,
        side,
        calibration,
        reference_frame,
        world_pose,
    )


def _target_source(
    backend,
    side,
    trajectory,
    plant,
    calibration,
    static_targets,
    look_at_object_motion,
):
    if not isinstance(static_targets, DualArmFramedTargets):
        raise TypeError("static_targets must be DualArmFramedTargets")
    states = frames.controller_states(plant, calibration)
    selected_state = states.for_arm(side)
    reference_frame = TargetFrame(trajectory.reference_frame)
    start_reference = _configured_start_pose(
        plant,
        side,
        calibration,
        reference_frame,
        trajectory,
        static_targets,
        selected_state.ee_pose_world,
    )
    look_at_object = None
    if (
        trajectory.orientation is not None
        and trajectory.orientation.policy == "look_at_sim_object"
    ):
        if look_at_object_motion is None:
            raise ValueError(
                "look_at_sim_object orientation requires "
                "[simulation.look_at_object_motion]"
            )
        if (
            trajectory.orientation.object_body
            != look_at_object_motion.body_name
        ):
            raise ValueError(
                "trajectory orientation object_body must match "
                "simulation.look_at_object_motion.body_name"
            )
        look_at_object = SimulatedMocapPointSource(
            backend, look_at_object_motion
        )
        look_at_object.apply(0.0)
    materialized = materialize_trajectory(
        trajectory,
        start_reference,
        world_point_source=look_at_object,
    )
    selected_source = materialized.source

    other = "right" if side == "left" else "left"
    other_hold = StaticTargetSource(static_targets.for_arm(other))
    source = (
        IndependentArmTargetSource(
            right=other_hold, left=selected_source
        )
        if side == "left"
        else IndependentArmTargetSource(
            right=selected_source, left=other_hold
        )
    )
    end_reference = (
        materialized.program.sample(
            materialized.duration_s
        ).pose
        if look_at_object is not None
        else selected_source.sample(
            materialized.duration_s
        ).pose
    )
    end_world = frames.resolve_target_world(
        plant,
        side,
        calibration,
        FramedTarget(reference_frame, end_reference, Twist.zero()),
    ).pose_world
    singular_values = np.linalg.svd(
        selected_state.jacobian_world, compute_uv=False
    )
    return (
        source,
        selected_source,
        start_reference,
        end_reference,
        selected_state.ee_pose_world,
        end_world,
        materialized.duration_s,
        materialized.boundary_times_s,
        materialized.rate_bounds,
        singular_values,
        _limited_joint_margin(
            backend,
            side,
            selected_state.joints.position_rad,
        ),
        look_at_object,
    )


def prepare_target_trajectory(
    backend,
    calibration,
    side,
    trajectory,
    initial_joint_position_rad,
    static_targets,
    look_at_object_motion=None,
):
    _set_initial_joint_posture(
        backend, side, initial_joint_position_rad
    )
    plant = backend.read_state(Twist.zero())
    (
        source,
        selected_source,
        start_reference,
        end_reference,
        start_world,
        end_world,
        duration_s,
        boundary_times_s,
        rate_bounds,
        singular_values,
        joint_margin,
        look_at_object,
    ) = _target_source(
        backend,
        side,
        trajectory,
        plant,
        calibration,
        static_targets,
        look_at_object_motion,
    )
    return TargetTrajectorySetup(
        source=source,
        selected_source=selected_source,
        start_pose_reference=start_reference,
        end_pose_reference=end_reference,
        start_pose_world=start_world,
        end_pose_world=end_world,
        duration_s=duration_s,
        boundary_times_s=boundary_times_s,
        rate_bounds=rate_bounds,
        initial_singular_values=singular_values,
        initial_joint_margin_rad=joint_margin,
        look_at_object=look_at_object,
    )


def print_target_trajectory_setup(side, trajectory, setup):
    print(f"Configured {side} target trajectory:")
    orientation = (
        "segment/static"
        if trajectory.orientation is None
        else trajectory.orientation.policy
    )
    print(
        f"reference_frame={trajectory.reference_frame} "
        f"start={trajectory.start} loop={trajectory.loop} "
        f"segments={[item.type for item in trajectory.segments]} "
        f"orientation={orientation} "
        f"duration_s={setup.duration_s:.6f}"
    )
    if trajectory.orientation is not None:
        policy = trajectory.orientation
        if policy.policy == "look_at_fixed_world_point":
            print(
                "look_at_object_world_m="
                f"{np.array2string(np.asarray(policy.object_position_world_m), precision=9)}"
            )
        else:
            print(f"look_at_sim_object_body={policy.object_body}")
        print(
            "tool_forward_axis="
            f"{np.array2string(np.asarray(policy.tool_forward_axis), precision=6)} "
            "tool_up_axis="
            f"{np.array2string(np.asarray(policy.tool_up_axis), precision=6)} "
            "world_up_direction="
            f"{np.array2string(np.asarray(policy.world_up_direction), precision=6)}"
        )
    print(
        "resolved_world_start_m="
        f"{np.array2string(setup.start_pose_world.position_m, precision=9)}"
    )
    print(
        "resolved_world_end_m="
        f"{np.array2string(setup.end_pose_world.position_m, precision=9)}"
    )
    print(
        "initial_jacobian_singular_values="
        f"{np.array2string(setup.initial_singular_values, precision=6)}"
    )
    print(
        "trajectory will execute in the real-time simulation loop; "
        "no preflight replay was run"
    )
