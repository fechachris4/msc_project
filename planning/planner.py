"""Build a planned target source from configuration and plant state.

This is the composition root of the planning layer.  It reads the same
``[targets.<arm>]`` goal the reactive baseline uses, captures the torso
pose to fix the collision frame, optimises the path, and wraps it in the
``PlannedDualArmSource`` the Runner already knows how to sample.

It refuses to build alongside an enabled cylinder keep-out router: that
router replaces the reference POSITION while passing the planner's TWIST
through unchanged (``controller/runner.py:201-204``), so the controller's
P and D terms would reference different points and the executed path
would be the router's, not the planner's.
"""

from dataclasses import dataclass, replace

import numpy as np

from controller import frames
from controller.cylinder_router import CylinderKeepout
from controller.runner import keepout_from_config
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
from planning.obstacles import Sphere, build_obstacle_set
from planning.path_optimizer import OptimizerParams, optimize_path
from planning.plan_source import (
    LeadCompensation,
    PlannedArmPath,
    PlannedDualArmSource,
    hold_source,
)
from runtime_config import CONFIG, PlanningConfig


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """One arm's plan plus the evidence it was checked."""

    side: str
    result: object
    goal_pose_world: Pose
    start_pose_world: Pose


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    """Everything a caller needs to run and to report the plan."""

    source: PlannedDualArmSource
    plans: tuple
    torso_pose_world: Pose

    def for_side(self, side):
        for plan in self.plans:
            if plan.side == side:
                return plan
        return None


def disabled_keepout(config=None):
    """The configured keep-out geometry, switched off.

    Planned runs must not go through the cylinder router: it replaces the
    reference POSITION while passing the planner's TWIST through unchanged,
    so the controller's P and D terms would reference different points.
    Flipping only ``enabled`` keeps the diagnostics describing the same
    cylinder.
    """
    source = CONFIG.cylinder_keepout if config is None else config
    return replace(keepout_from_config(source), enabled=False)


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


def build_obstacles_for(plant, planning_config, other_arm_sphere=None):
    extra = () if other_arm_sphere is None else (other_arm_sphere,)
    return build_obstacle_set(
        CONFIG.human_safety,
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


def plan_from_state(
    plant,
    calibration,
    planning_config=None,
    reactive_pose_config=None,
    cylinder_keepout=None,
    elapsed_time_s=0.0,
):
    """Optimise a path per planned arm and wrap it for the Runner."""
    planning_config = (
        CONFIG.planning if planning_config is None else planning_config
    )
    reactive_pose_config = (
        CONFIG.reactive_pose
        if reactive_pose_config is None else reactive_pose_config
    )
    if not isinstance(plant, PlantState):
        raise TypeError("plant must be a PlantState")
    if not isinstance(calibration, MountCalibration):
        raise TypeError("calibration must be a MountCalibration")
    if not isinstance(planning_config, PlanningConfig):
        raise TypeError("planning_config must be a PlanningConfig")
    # Resolve to the LIVE configuration when omitted, so the refusal
    # this module advertises actually fires on the default call path
    # (the shipped config enables the router).
    if cylinder_keepout is None:
        cylinder_keepout = keepout_from_config(CONFIG.cylinder_keepout)
    if isinstance(cylinder_keepout, CylinderKeepout):
        if cylinder_keepout.enabled:
            raise ValueError(
                "the cylinder keep-out router rewrites planned reference "
                "positions while passing the planned twist through "
                "unchanged; disable [cylinder_keepout] to plan"
            )
    else:
        raise TypeError("cylinder_keepout must be a CylinderKeepout")
    elapsed_time_s = float(elapsed_time_s)
    if not np.isfinite(elapsed_time_s) or elapsed_time_s < 0.0:
        raise ValueError("elapsed_time_s must be finite and non-negative")

    states = frames.controller_states(plant, calibration)
    lead = LeadCompensation.from_config(
        reactive_pose_config,
        enabled=planning_config.lead_compensation_enabled,
    )
    sides = planned_sides(planning_config)
    limits = trajectory_limits(planning_config)
    params = optimizer_params(planning_config)

    paths = {}
    plans = []
    for side in ("right", "left"):
        start_pose = states.for_arm(side).ee_pose_world
        if side not in sides:
            paths[side] = PlannedArmPath(hold_source(start_pose), lead)
            continue
        goal = goal_pose_world(
            plant, side, calibration, CONFIG.target(side)
        )
        obstacles = build_obstacles_for(plant, planning_config)
        result = optimize_path(
            start_pose,
            goal,
            plant.torso_pose_world,
            obstacles,
            limits,
            params,
        )
        paths[side] = PlannedArmPath(
            result.trajectory, lead, float(elapsed_time_s)
        )
        plans.append(
            ArmPlan(
                side=side,
                result=result,
                goal_pose_world=goal,
                start_pose_world=start_pose,
            )
        )

    return PlanningOutcome(
        source=PlannedDualArmSource(paths["right"], paths["left"]),
        plans=tuple(plans),
        torso_pose_world=plant.torso_pose_world,
    )


def remaining_clearance(outcome, plant, planning_config, samples=40):
    """Torso-frame clearance of the plan under the CURRENT torso pose.

    The plan is world-frame and therefore does not move with the wearer.
    This is the quantity that decays as the torso moves, and it is the
    signal a replan trigger should watch.
    """
    obstacles = build_obstacles_for(plant, planning_config)
    rotation = plant.torso_pose_world.rotation.T
    origin = plant.torso_pose_world.position_m
    worst = np.inf
    for plan in outcome.plans:
        trajectory = outcome.source.for_arm(plan.side).source
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
