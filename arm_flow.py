"""Per-arm reference composition root.

One reusable single-arm flow, run independently for right and left:

    intent ([targets.<arm>] goal | [targets.<arm>.trajectory] segments)
      -> nominal geometric path      (controller/trajectory_config.py)
      -> optional feasibility stage  (planning/planner.py optimiser, or the
                                      cylinder router as a waypoint-producing
                                      path transformation)
      -> shared timing + limits      (controller/trajectory.py)
      -> shared orientation step     (waypoint interpolation / look-at)
      -> optional lead conditioning  (controller/lead_compensation.py)
      -> one single-arm TargetSource

Each arm is built from its OWN configuration and the same initial plant
snapshot; no stage can see, hold, or replace the other arm's reference.
The two flows meet only in ``DualArmFlow.source``, the thin dual-arm
container the Runner samples once per cycle.

Cylinder keep-out routing happens HERE, before timing, so a routed
reach's desired position and velocity come from the same timed path.  The
runner no longer rewrites sampled targets.
"""

from dataclasses import dataclass

import numpy as np

from controller import frames
from controller.cylinder_router import (
    CylinderKeepout,
    CylinderRouteKind,
    CylinderRouter,
    route_kind_name,
)
from controller.state import Pose, TargetFrame, Twist
from controller.trajectory import (
    IndependentArmTargetSource,
    StaticTargetSource,
    timed_waypoint_trajectory,
    _rotation_from_vector,
    _rotation_vector,
)
from planning import planner
from runtime_config import CONFIG, CylinderKeepoutConfig
from sim.target_trajectory import (
    apply_initial_postures,
    prepare_target_trajectory,
)

SIDES = ("right", "left")


def keepout_from_config(config):
    """Build the shared world-frame keep-out from the TOML config record.

    The field-for-field copy remains comparable with the hardware schema, but
    this simulation interprets every geometric value in WORLD coordinates.
    """
    if not isinstance(config, CylinderKeepoutConfig):
        raise TypeError("config must be a CylinderKeepoutConfig")
    return CylinderKeepout(
        enabled=config.cylinder_keepout_enabled,
        center_xy_m=(
            config.cylinder_keepout_center_x_m,
            config.cylinder_keepout_center_y_m,
        ),
        radius_m=config.cylinder_keepout_radius_m,
        z_min_m=config.cylinder_keepout_z_min_m,
        z_max_m=config.cylinder_keepout_z_max_m,
        clearance_m=config.cylinder_keepout_clearance_m,
        waypoint_tolerance_m=config.cylinder_waypoint_tolerance_m,
    )


@dataclass(frozen=True, slots=True)
class RouteReport:
    """One composed keep-out route, retained for display and analysis."""

    kind: str
    waypoints_world_m: tuple
    requested_target_world_m: np.ndarray
    effective_target_world_m: np.ndarray
    target_adjusted: bool


@dataclass(frozen=True, slots=True)
class ArmFlow:
    """One arm's composed reference and the artefacts that produced it."""

    side: str
    kind: str  # "static" | "trajectory" | "planned" | "routed_reach"
    source: object
    duration_s: float
    plan: object = None
    route: RouteReport | None = None
    trajectory_setup: object = None

    @property
    def look_at_object(self):
        if self.trajectory_setup is None:
            return None
        return self.trajectory_setup.look_at_object


@dataclass(frozen=True, slots=True)
class DualArmFlow:
    """The thin two-arm layer: two independent flows, one sampled source."""

    right: ArmFlow
    left: ArmFlow
    source: IndependentArmTargetSource

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")

    @property
    def flows(self):
        return (self.right, self.left)

    @property
    def plans(self):
        return tuple(
            flow.plan for flow in self.flows if flow.plan is not None
        )

    @property
    def routes(self):
        return {
            flow.side: flow.route
            for flow in self.flows
            if flow.route is not None
        }

    @property
    def look_at_objects(self):
        seen = []
        for flow in self.flows:
            obj = flow.look_at_object
            if obj is not None and all(obj is not item for item in seen):
                seen.append(obj)
        return tuple(seen)

    @property
    def duration_s(self):
        return max(flow.duration_s for flow in self.flows)


def _routed_reach(side, plant, calibration, static_target, keepout, limits):
    """Route one static-goal reach around the keep-out, then time it.

    Returns ``None`` when no routing is needed, keeping today's held-target
    semantics for clear reaches.  When routing fires, the goal is resolved
    to WORLD at composition time (a torso-frame goal is frozen, exactly as
    a planned goal is) and the detour becomes an ordinary timed waypoint
    trajectory: position and twist come from the same path.
    """
    states = frames.controller_states(plant, calibration)
    start_pose = states.for_arm(side).ee_pose_world
    goal_pose = frames.resolve_target_world(
        plant, side, calibration, static_target
    ).pose_world
    route = CylinderRouter(keepout).plan(
        start_pose.position_m, goal_pose.position_m
    )
    if route.kind is CylinderRouteKind.DIRECT and not route.target_adjusted:
        return None

    positions = [start_pose.position_m] + [
        np.asarray(point, dtype=float) for point in route.waypoints
    ]
    # The same orientation step the optimiser's build uses: interpolate
    # start->goal along the waypoint sequence via the shared SO(3) log/exp.
    relative = _rotation_vector(
        start_pose.rotation.T @ goal_pose.rotation
    )
    fractions = np.linspace(0.0, 1.0, len(positions))
    poses = tuple(
        Pose(
            position,
            start_pose.rotation
            @ _rotation_from_vector(fraction * relative),
        )
        for position, fraction in zip(positions, fractions)
    )
    source = timed_waypoint_trajectory(
        TargetFrame.WORLD, poses, None, limits
    )
    report = RouteReport(
        kind=route_kind_name(route.kind),
        waypoints_world_m=tuple(route.waypoints),
        requested_target_world_m=route.requested_target,
        effective_target_world_m=route.effective_target,
        target_adjusted=route.target_adjusted,
    )
    return ArmFlow(
        side=side,
        kind="routed_reach",
        source=source,
        duration_s=source.duration_s,
        route=report,
    )


def build_arm_flow(
    backend,
    calibration,
    side,
    plant,
    static_targets,
    keepout,
    config=CONFIG,
    controlled=True,
):
    """Compose ONE arm's reference from its own configuration only."""
    trajectory = config.target(side).trajectory
    planned = (
        config.planning.enabled
        and side in planner.planned_sides(config.planning)
    )
    if trajectory is not None and planned:
        raise ValueError(
            f"[planning] and [targets.{side}.trajectory] both drive the "
            f"{side} arm; disable one"
        )

    if trajectory is not None:
        setup = prepare_target_trajectory(
            backend,
            calibration,
            side,
            trajectory,
            plant,
            static_targets,
            config.simulation.look_at_object_motion,
        )
        return ArmFlow(
            side=side,
            kind="trajectory",
            source=setup.source,
            duration_s=setup.duration_s,
            trajectory_setup=setup,
        )

    if planned:
        plan = planner.plan_arm(
            plant,
            calibration,
            side,
            config.planning,
            config.reactive_pose,
            target_config=config.target(side),
            human_safety_config=config.human_safety,
        )
        if not plan.result.success:
            raise RuntimeError(
                f"{side} planner rejected its path: "
                f"{plan.result.message}; "
                f"collision_free={plan.result.collision_free}, "
                f"margin_met={plan.result.margin_met}, "
                f"final_clearance_m="
                f"{plan.result.final_min_clearance_m:.6f}"
            )
        return ArmFlow(
            side=side,
            kind="planned",
            source=plan.source,
            duration_s=plan.result.duration_s,
            plan=plan,
        )

    static_target = static_targets.for_arm(side)
    if keepout.enabled and controlled:
        routed = _routed_reach(
            side,
            plant,
            calibration,
            static_target,
            keepout,
            planner.trajectory_limits(config.planning),
        )
        if routed is not None:
            return routed
    return ArmFlow(
        side=side,
        kind="static",
        source=StaticTargetSource(static_target),
        duration_s=0.0,
    )


def build_flows(
    backend,
    calibration,
    static_targets,
    arms=SIDES,
    config=CONFIG,
):
    """Build both arms' flows from one shared initial snapshot.

    Postures are applied once for every trajectory arm BEFORE the single
    ``read_state`` snapshot, so both flows compose against the same
    initial state and neither build can invalidate the other's.
    """
    trajectory_sides = tuple(
        side for side in SIDES
        if config.target(side).trajectory is not None
    )
    if trajectory_sides:
        apply_initial_postures(
            backend,
            {
                side: config.simulation.initial_joint_position(side)
                for side in trajectory_sides
            },
        )
    plant = backend.read_state(Twist.zero())
    keepout = keepout_from_config(config.cylinder_keepout)
    flows = {
        side: build_arm_flow(
            backend,
            calibration,
            side,
            plant,
            static_targets,
            keepout,
            config,
            controlled=side in arms,
        )
        for side in SIDES
    }
    return DualArmFlow(
        right=flows["right"],
        left=flows["left"],
        source=IndependentArmTargetSource(
            right=flows["right"].source,
            left=flows["left"].source,
        ),
    )
