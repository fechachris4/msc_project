"""Explicit reactive-pose to joint-position control pipeline.

This module contains controller and actuation composition only. It does not
read MuJoCo state, transform frames, advance a clock, or write actuators.
"""

from dataclasses import dataclass

import numpy as np

from controller.position_actuation import (
    PositionActuationLimits,
    PositionIntegrator,
)
from controller.reactive_controller import JointCentering, ReactiveController
from controller.state import (
    DualArmControllerStates,
    DualArmWorldTargets,
    JointPositionCommand,
    PlantState,
)
from runtime_config import CONFIG, ReactivePoseConfig


CONTROL = CONFIG.reactive_pose
LIMITS = CONFIG.limits
ARMS = ("right", "left")


def _read_only_copy(value):
    copied = np.array(value, copy=True)
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(
        copied.shape
    )


@dataclass(frozen=True, slots=True)
class ControlTrace:
    """Read-only snapshots of one arm's control stages for one cycle."""

    J: np.ndarray
    e_pos: np.ndarray
    e_rot: np.ndarray
    e_v: np.ndarray
    e_w: np.ndarray
    p_twist: np.ndarray
    d_twist: np.ndarray
    task_twist: np.ndarray
    q: np.ndarray
    qdot_measured: np.ndarray
    qdot_raw: np.ndarray
    qdot_speed_clipped: np.ndarray
    qdot_effective: np.ndarray
    ctrl_before: np.ndarray
    ctrl_after: np.ndarray
    speed_saturated: np.ndarray
    lead_clamped: np.ndarray
    range_clamped: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _read_only_copy(getattr(self, name))
            )


@dataclass(frozen=True, slots=True)
class DualArmControlTraces:
    """Fixed dual-arm trace record; an unselected arm is ``None``."""

    right: ControlTrace | None
    left: ControlTrace | None

    def __post_init__(self):
        for side in ARMS:
            value = getattr(self, side)
            if value is not None and not isinstance(value, ControlTrace):
                raise TypeError(f"{side} must be ControlTrace or None")

    def __getitem__(self, side):
        if side not in ARMS:
            raise KeyError(side)
        value = getattr(self, side)
        if value is None:
            raise KeyError(side)
        return value

    def __iter__(self):
        return (side for side in ARMS if getattr(self, side) is not None)

    def items(self):
        return tuple((side, self[side]) for side in self)

    def values(self):
        return tuple(self[side] for side in self)


@dataclass(frozen=True, slots=True)
class ArmPipelineSetup:
    """Fixed kinematic/actuator facts needed by one arm pipeline."""

    centering: JointCentering
    actuation_limits: PositionActuationLimits

    def __post_init__(self):
        if not isinstance(self.centering, JointCentering):
            raise TypeError("centering must be JointCentering")
        if not isinstance(self.actuation_limits, PositionActuationLimits):
            raise TypeError(
                "actuation_limits must be PositionActuationLimits"
            )


@dataclass(frozen=True, slots=True)
class DualArmPipelineSetup:
    right: ArmPipelineSetup
    left: ArmPipelineSetup

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")


class ReactivePositionPipeline:
    """Stateful pose-control pipeline; reset means reconstruct this object."""

    def __init__(
        self,
        plant_state,
        setup,
        controller_config=CONTROL,
    ):
        if not isinstance(plant_state, PlantState):
            raise TypeError("plant_state must be a PlantState")
        if not isinstance(setup, DualArmPipelineSetup):
            raise TypeError("setup must be a DualArmPipelineSetup")
        if not isinstance(controller_config, ReactivePoseConfig):
            raise TypeError("controller_config must be ReactivePoseConfig")

        self._controllers = {}
        self._integrators = {}
        for side in ARMS:
            arm_setup = setup.for_arm(side)
            self._controllers[side] = ReactiveController(
                controller_config, arm_setup.centering
            )
            self._integrators[side] = PositionIntegrator(
                plant_state.arm(side).position_rad,
                arm_setup.actuation_limits,
            )

    def command(self):
        """Return the current persistent position command for both arms."""
        return JointPositionCommand(
            right_position_rad=self._integrators["right"].command_rad,
            left_position_rad=self._integrators["left"].command_rad,
        )

    def step(self, states, targets, dt_s, arms=ARMS):
        """Compute one command without reading or writing a plant backend."""
        if not isinstance(states, DualArmControllerStates):
            raise TypeError("states must be DualArmControllerStates")
        if not isinstance(targets, DualArmWorldTargets):
            raise TypeError("targets must be DualArmWorldTargets")
        selected = tuple(arms)
        if len(set(selected)) != len(selected):
            raise ValueError("arms must not contain duplicates")
        for side in selected:
            if side not in ARMS:
                raise ValueError(f"unknown arm: {side!r}")

        traces = {"right": None, "left": None}
        for side in selected:
            state = states.for_arm(side)
            output = self._controllers[side].compute(
                state, targets.for_arm(side)
            )
            actuation = self._integrators[side].step(
                state.joints.position_rad,
                output.solve.qdot_raw,
                dt_s,
            )
            traces[side] = ControlTrace(
                J=state.jacobian_world,
                e_pos=output.e_pos,
                e_rot=output.e_rot,
                e_v=output.e_v,
                e_w=output.e_w,
                p_twist=output.solve.p_twist,
                d_twist=output.solve.d_twist,
                task_twist=output.solve.task_twist,
                q=state.joints.position_rad,
                qdot_measured=state.joints.velocity_rad_s,
                qdot_raw=output.solve.qdot_raw,
                qdot_speed_clipped=actuation.qdot_speed_clipped,
                qdot_effective=actuation.qdot_effective,
                ctrl_before=actuation.command_before_rad,
                ctrl_after=actuation.command_after_rad,
                speed_saturated=actuation.speed_saturated,
                lead_clamped=actuation.lead_clamped,
                range_clamped=actuation.range_clamped,
            )
        return self.command(), DualArmControlTraces(**traces)
