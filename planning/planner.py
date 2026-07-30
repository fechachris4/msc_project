"""Plan one arm's collision-aware path from configuration and plant state.

The planner is a feasibility stage inside ONE arm's reference pipeline.
It reads that arm's ``[targets.<arm>]`` goal, captures the torso pose to
fix the collision frame, optimises the path, and returns the optimised
trajectory plus the evidence it was checked.  The optimised trajectory is
built by the shared trajectory layer (``timed_waypoint_trajectory`` with
the shared orientation interpolation, ``planning/path_optimizer.py``), so
a planned path re-enters the same timing, orientation, sampling, and
controller pipeline as every other motion.

Delivery to the controller happens through the ordinary single-arm
``TargetSource`` seam in the composition root (``arm_flow.py``).  The
planner never sees, holds, or replaces the other arm's reference, and it
does not interact with the cylinder keep-out router: routing is a
composition-time path transformation for unplanned reaches, not a runtime
rewrite, so the two can no longer corrupt each other's references.
"""

from dataclasses import dataclass

import numpy as np

from controller import frames
from controller.lead_compensation import (
    LeadCompensatedSource,
    LeadCompensation,
)
from controller.state import (
    FramedTarget,
    MountCalibration,
    PlantState,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import TrajectoryLimits
from controller.transforms import rotation_from_rpy
from planning.obstacles import build_obstacle_set
from planning.path_optimizer import OptimizerParams, optimize_path
from runtime_config import CONFIG, PlanningConfig


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """One arm's plan, its delivered source, and the evidence it was checked.

    ``source`` is the reference actually delivered to the controller
    (lead-conditioned when configured); ``result.trajectory`` is the
    optimised path BEFORE conditioning — the comparison reference for
    analysis and visualisation.
    """

    side: str
    result: object
    goal_pose_world: Pose
    start_pose_world: Pose
    torso_pose_world: Pose
    source: object
    lead: LeadCompensation


def planned_sides(planning_config):
    if not isinstance(planning_config, PlanningConfig):
        raise TypeError("planning_config must be a PlanningConfig")
    if planning_config.arm == "both":
        return ("right", "left")
    return (planning_config.arm,)


def optimizer_params(planning_config):
    return OptimizerParams(
        waypoint_count=planning_config.waypoint_count,
        dense_samples=planning_config.dense_samples,
        clearance_margin_m=planning_config.clearance_margin_m,
        smoothness_weight=planning_config.smoothness_weight,
        obstacle_weight=planning_config.obstacle_weight,
        max_iterations=planning_config.max_iterations,
        tool_radius_m=planning_config.tool_radius_m,
        deviation_weight=planning_config.deviation_weight,
        reach_allowance_m=planning_config.reach_allowance_m,
    )


def trajectory_limits(planning_config):
    return TrajectoryLimits(
        planning_config.max_linear_speed_m_s,
        planning_config.max_linear_acceleration_m_s2,
        planning_config.max_angular_speed_rad_s,
        planning_config.max_angular_acceleration_rad_s2,
    )


def goal_pose_world(plant, side, calibration, target_config):
    """Resolve the existing ``[targets.<arm>]`` goal into world coordinates."""
    framed = FramedTarget(
        TargetFrame(target_config.reference_frame),
        Pose(
            np.asarray(target_config.position_m, dtype=float),
            rotation_from_rpy(target_config.rpy_rad),
        ),
        Twist.zero(),
    )
    return frames.resolve_target_world(
        plant, side, calibration, framed
    ).pose_world


def build_obstacles_for(
    plant, planning_config, other_arm_sphere=None, human_safety_config=None
):
    extra = () if other_arm_sphere is None else (other_arm_sphere,)
    return build_obstacle_set(
        CONFIG.human_safety
        if human_safety_config is None else human_safety_config,
        plant.torso_pose_world,
        floor_height_world_m=(
            planning_config.floor_height_world_m
            if planning_config.include_floor else None
        ),
        torso_box_half_extent_m=(
            planning_config.torso_box_half_extent_m
            if planning_config.include_torso_box else None
        ),
        extra_spheres=extra,
    )


def plan_arm(
    plant,
    calibration,
    side,
    planning_config=None,
    reactive_pose_config=None,
    target_config=None,
    human_safety_config=None,
):
    """Optimise one arm's path and package it for per-arm composition.

    Every configuration record can be injected so a caller composing from
    a non-global config (tests, analysis sweeps) gets a plan consistent
    with the rest of its flow; omitted records fall back to ``CONFIG``.
    """
    planning_config = (
        CONFIG.planning if planning_config is None else planning_config
    )
    reactive_pose_config = (
        CONFIG.reactive_pose
        if reactive_pose_config is None else reactive_pose_config
    )
    if target_config is None:
        target_config = CONFIG.target(side)
    if not isinstance(plant, PlantState):
        raise TypeError("plant must be a PlantState")
    if not isinstance(calibration, MountCalibration):
        raise TypeError("calibration must be a MountCalibration")
    if not isinstance(planning_config, PlanningConfig):
        raise TypeError("planning_config must be a PlanningConfig")
    if side not in ("right", "left"):
        raise ValueError(f"unknown arm: {side!r}")

    states = frames.controller_states(plant, calibration)
    start_pose = states.for_arm(side).ee_pose_world
    goal = goal_pose_world(plant, side, calibration, target_config)
    obstacles = build_obstacles_for(
        plant, planning_config, human_safety_config=human_safety_config
    )
    result = optimize_path(
        start_pose,
        goal,
        plant.torso_pose_world,
        obstacles,
        trajectory_limits(planning_config),
        optimizer_params(planning_config),
    )
    lead = LeadCompensation.from_config(
        reactive_pose_config,
        enabled=planning_config.lead_compensation_enabled,
    )
    source = (
        LeadCompensatedSource(result.trajectory, lead)
        if lead.enabled
        else result.trajectory
    )
    return ArmPlan(
        side=side,
        result=result,
        goal_pose_world=goal,
        start_pose_world=start_pose,
        torso_pose_world=plant.torso_pose_world,
        source=source,
        lead=lead,
    )


def remaining_clearance(plans, plant, planning_config, samples=40):
    """Torso-frame clearance of the plans under the CURRENT torso pose.

    A plan is world-frame and therefore does not move with the wearer.
    This is the quantity that decays as the torso moves, and it is the
    signal a replan trigger should watch.
    """
    obstacles = build_obstacles_for(plant, planning_config)
    rotation = plant.torso_pose_world.rotation.T
    origin = plant.torso_pose_world.position_m
    worst = np.inf
    for plan in plans:
        trajectory = plan.result.trajectory
        times = np.linspace(0.0, trajectory.duration_s, samples)
        points_world = np.stack([
            trajectory.sample(float(t)).pose.position_m for t in times
        ])
        points_torso = (points_world - origin) @ rotation.T
        distance, _ = obstacles.distance_and_gradient(points_torso)
        worst = min(
            worst, float(np.min(distance) - planning_config.tool_radius_m)
        )
    return worst
