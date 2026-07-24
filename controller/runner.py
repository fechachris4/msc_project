"""Explicit runner for the current reactive pose-to-position pipeline."""

from dataclasses import dataclass

import numpy as np

from controller import frames
from controller.backend import PlantBackend
from controller.cylinder_router import (
    CylinderKeepout,
    CylinderRouteFollower,
    route_kind_name,
)
from controller.servo import (
    DualArmControlTraces,
    DualArmPipelineSetup,
    ReactivePositionPipeline,
)
from controller.state import (
    DualArmControllerStates,
    DualArmFramedTargets,
    DualArmWorldTargets,
    JointPositionCommand,
    MountCalibration,
    PlantState,
    Pose,
    WorldTarget,
)
from controller.trajectory import (
    as_dual_arm_target_source,
    sample_dual_arm_target_source,
)
from runtime_config import CONFIG, CylinderKeepoutConfig, ReactivePoseConfig


def keepout_from_config(config):
    """Build the router's keep-out from the TOML config record.

    Mirrors ``basic_control/src/app/main.cpp`` lines 194-204, which performs
    the same field-for-field copy from EffectiveConfig into CylinderKeepout.
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
class CylinderRouteStatus:
    """Per-arm routing diagnostics; the C++ ControllerStatus route_* fields."""

    kind: str
    waypoint_count: int
    waypoint_index: int
    at_final_waypoint: bool
    target_adjusted: bool
    route_changed: bool
    requested_target_base_m: np.ndarray
    effective_target_base_m: np.ndarray
    active_waypoint_world_m: np.ndarray
    waypoints_world_m: tuple


@dataclass(frozen=True, slots=True)
class RunnerCycle:
    """One completed exchange, retaining both sides of the boundary."""

    input_state: PlantState
    target_elapsed_time_s: float
    sampled_targets: DualArmFramedTargets
    resolved_targets: DualArmWorldTargets
    routed_targets: DualArmWorldTargets
    controller_states: DualArmControllerStates
    command: JointPositionCommand
    traces: DualArmControlTraces
    next_state: PlantState
    cylinder_routes: dict


class ReactivePositionRunner:
    """Own cycle ordering; backend and controller never inspect each other."""

    def __init__(
        self,
        backend,
        calibration,
        pipeline_setup,
        source_targets,
        arms=("right", "left"),
        controller_config=CONFIG.reactive_pose,
        cylinder_keepout=CONFIG.cylinder_keepout,
    ):
        if not isinstance(backend, PlantBackend):
            raise TypeError("backend must satisfy PlantBackend")
        if not isinstance(calibration, MountCalibration):
            raise TypeError("calibration must be MountCalibration")
        if not isinstance(pipeline_setup, DualArmPipelineSetup):
            raise TypeError("pipeline_setup must be DualArmPipelineSetup")
        if not isinstance(controller_config, ReactivePoseConfig):
            raise TypeError("controller_config must be ReactivePoseConfig")
        selected = tuple(arms)
        if not selected or any(
            side not in ("right", "left") for side in selected
        ):
            raise ValueError("arms must be a non-empty subset of right/left")
        if len(set(selected)) != len(selected):
            raise ValueError("arms must not contain duplicates")

        self._backend = backend
        self._calibration = calibration
        self._pipeline_setup = pipeline_setup
        self._target_source = as_dual_arm_target_source(source_targets)
        self._arms = selected
        self._controller_config = controller_config
        self._plant_state = None
        self._pipeline = None
        self._target_time_origin_s = None
        self._keepout = (
            cylinder_keepout
            if isinstance(cylinder_keepout, CylinderKeepout)
            else keepout_from_config(cylinder_keepout)
        )
        self._followers = {
            side: CylinderRouteFollower(self._keepout) for side in selected
        }
        # The C++ replans on a TargetStore sequence bump. The analogue here is
        # a change in the SAMPLED FRAMED target: frame-relative, so torso
        # motion alone never counts as the operator issuing a new target.
        self._accepted_target = {side: None for side in selected}

    @property
    def cylinder_keepout(self):
        return self._keepout

    def _base_pose_world(self, plant, side):
        """T_W_B for one arm — the frame the keep-out cylinder is defined in."""
        return frames.compose_pose(
            plant.torso_pose_world, self._calibration.for_arm(side)
        )

    def _accepted_target_key(self, sampled_targets, side):
        target = sampled_targets.for_arm(side)
        return (
            target.reference_frame,
            tuple(np.asarray(target.pose.position_m, dtype=float).tolist()),
        )

    def _route_targets(
        self, plant, sampled_targets, resolved, controller_states
    ):
        """Substitute the active waypoint for each arm's world target position.

        Positions cross into the arm base frame, through the router, and back.
        Orientation and twist are passed through untouched, so intermediate
        waypoints are followed at the REQUESTED orientation. When the keep-out
        is disabled this returns ``resolved`` unchanged, preserving the
        simulation's original direct-target behaviour exactly.
        """
        if not self._keepout.enabled:
            return resolved, {}

        routed = {"right": resolved.right, "left": resolved.left}
        statuses = {}
        for side in self._arms:
            base_pose = self._base_pose_world(plant, side)
            rotation = base_pose.rotation
            origin = base_pose.position_m
            follower = self._followers[side]

            ee_world = controller_states.for_arm(side).ee_pose_world.position_m
            ee_base = rotation.T @ (ee_world - origin)

            target = resolved.for_arm(side)
            target_base = rotation.T @ (target.pose_world.position_m - origin)

            key = self._accepted_target_key(sampled_targets, side)
            route_changed = key != self._accepted_target[side]
            if route_changed:
                self._accepted_target[side] = key
                follower.set_target(ee_base, target_base)

            waypoint_base = follower.update(ee_base)
            waypoint_world = origin + rotation @ waypoint_base
            routed[side] = WorldTarget(
                Pose(waypoint_world, target.pose_world.rotation),
                target.twist_world,
            )
            route = follower.route
            statuses[side] = CylinderRouteStatus(
                kind=route_kind_name(route.kind),
                waypoint_count=route.size,
                waypoint_index=follower.index,
                at_final_waypoint=follower.at_final_waypoint(),
                target_adjusted=route.target_adjusted,
                route_changed=route_changed,
                requested_target_base_m=route.requested_target,
                effective_target_base_m=route.effective_target,
                active_waypoint_world_m=waypoint_world,
                waypoints_world_m=tuple(
                    origin + rotation @ point for point in route.waypoints
                ),
            )
        return (
            DualArmWorldTargets(right=routed["right"], left=routed["left"]),
            statuses,
        )

    @property
    def current_state(self):
        if self._plant_state is None:
            raise RuntimeError("runner has not taken over the backend")
        return self._plant_state

    def start(self):
        if self._plant_state is not None:
            raise RuntimeError("runner is already started")
        plant_state = self._backend.takeover()
        try:
            self._pipeline = ReactivePositionPipeline(
                plant_state,
                self._pipeline_setup,
                self._controller_config,
            )
        except Exception:
            self._backend.release()
            raise
        self._plant_state = plant_state
        self._target_time_origin_s = plant_state.sample_time_s
        if self._keepout.enabled:
            # C++ Reset(): seed each follower at the measured pose so the very
            # first Update has a valid single-waypoint route.
            states = frames.controller_states(plant_state, self._calibration)
            for side in self._arms:
                base_pose = self._base_pose_world(plant_state, side)
                ee_world = states.for_arm(side).ee_pose_world.position_m
                self._followers[side].reset(
                    base_pose.rotation.T @ (ee_world - base_pose.position_m)
                )
                self._accepted_target[side] = None
        return plant_state

    def cycle(self, source_targets=None):
        if self._plant_state is None or self._pipeline is None:
            raise RuntimeError("runner must be started before cycling")
        input_state = self._plant_state
        target_elapsed = (
            input_state.sample_time_s - self._target_time_origin_s
        )
        target_source = (
            self._target_source
            if source_targets is None
            else as_dual_arm_target_source(source_targets)
        )
        sampled_targets = sample_dual_arm_target_source(
            target_source, target_elapsed
        )
        resolved = frames.resolve_targets_world(
            input_state, self._calibration, sampled_targets)
        controller_states = frames.controller_states(
            input_state, self._calibration)
        routed, cylinder_routes = self._route_targets(
            input_state, sampled_targets, resolved, controller_states)
        command, traces = self._pipeline.step(
            controller_states,
            routed,
            input_state.nominal_dt_s,
            self._arms,
        )
        next_state = self._backend.exchange(command)
        self._plant_state = next_state
        return RunnerCycle(
            input_state=input_state,
            target_elapsed_time_s=target_elapsed,
            sampled_targets=sampled_targets,
            resolved_targets=resolved,
            routed_targets=routed,
            controller_states=controller_states,
            command=command,
            traces=traces,
            next_state=next_state,
            cylinder_routes=cylinder_routes,
        )

    def close(self):
        if self._plant_state is None:
            return
        try:
            self._backend.release()
        finally:
            self._plant_state = None
            self._pipeline = None
            self._target_time_origin_s = None
