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
from controller.reactive_controller import (
    JointCentering,
    ReactiveController,
    SafetyVelocityProjector,
    SafetyVelocitySolve,
    constrain_velocity_for_human,
)
from controller.state import (
    DualArmControllerStates,
    DualArmHumanSafetyStates,
    DualArmWorldTargets,
    JointPositionCommand,
    PlantState,
)
from controller.link_spheres import LINK_SPHERES
from runtime_config import CONFIG, HumanSafetyConfig, ReactivePoseConfig


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
    ff_twist: np.ndarray
    task_twist: np.ndarray
    q: np.ndarray
    qdot_measured: np.ndarray
    qdot_raw: np.ndarray
    qdot_speed_clipped: np.ndarray
    qdot_safety_filtered: np.ndarray
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


class ArmControlPipeline:
    """ONE arm's complete control stage: controller, safety, actuation.

    This is the reusable single-arm control unit.  It never sees the
    other arm; the dual-arm wrapper below owns only command combination.
    """

    def __init__(
        self,
        initial_joint_position_rad,
        arm_setup,
        controller_config,
        human_safety_config,
        max_human_constraints,
    ):
        if not isinstance(arm_setup, ArmPipelineSetup):
            raise TypeError("arm_setup must be an ArmPipelineSetup")
        self._controller = ReactiveController(
            controller_config, arm_setup.centering
        )
        self._integrator = PositionIntegrator(
            initial_joint_position_rad,
            arm_setup.actuation_limits,
        )
        self._actuation_limits = arm_setup.actuation_limits
        self._human_safety_config = human_safety_config
        self._safety_projector = SafetyVelocityProjector(
            max_human_constraints, human_safety_config
        )

    @property
    def command_rad(self):
        return self._integrator.command_rad

    def step(self, state, target, dt_s, human_safety_state=None):
        """One arm's control cycle: world target in, trace and safety out."""
        output = self._controller.compute(state, target)
        if human_safety_state is None:
            # Direct pipeline users that did not supply evaluated geometry
            # retain the exact pre-safety actuation path.
            safety = SafetyVelocitySolve(
                qdot_safe=output.solve.qdot_raw,
                minimum_clearance_m=float("inf"),
                active_constraint_count=0,
                projection_iterations=0,
                max_constraint_violation_m_s=0.0,
                human_adjusted=False,
                limit_adjusted=False,
                stopped=False,
                reason="not_evaluated",
                limiting_points=(),
            )
        else:
            lower_velocity, upper_velocity = (
                self._integrator.velocity_bounds(
                    state.joints.position_rad, dt_s
                )
            )
            safety = constrain_velocity_for_human(
                output.solve.qdot_raw,
                lower_velocity,
                upper_velocity,
                human_safety_state,
                self._human_safety_config,
                measured_velocity_rad_s=(
                    state.joints.velocity_rad_s
                ),
                projector=self._safety_projector,
            )
        actuation = self._integrator.step(
            state.joints.position_rad,
            safety.qdot_safe,
            dt_s,
        )
        velocity_limit = self._actuation_limits.velocity_rad_s
        qdot_speed_clipped = np.clip(
            output.solve.qdot_raw,
            -velocity_limit,
            velocity_limit,
        )
        trace = ControlTrace(
            J=state.jacobian_world,
            e_pos=output.e_pos,
            e_rot=output.e_rot,
            e_v=output.e_v,
            e_w=output.e_w,
            p_twist=output.solve.p_twist,
            d_twist=output.solve.d_twist,
            ff_twist=output.solve.ff_twist,
            task_twist=output.solve.task_twist,
            q=state.joints.position_rad,
            qdot_measured=state.joints.velocity_rad_s,
            qdot_raw=output.solve.qdot_raw,
            qdot_speed_clipped=qdot_speed_clipped,
            qdot_safety_filtered=safety.qdot_safe,
            qdot_effective=actuation.qdot_effective,
            ctrl_before=actuation.command_before_rad,
            ctrl_after=actuation.command_after_rad,
            speed_saturated=(
                output.solve.qdot_raw != qdot_speed_clipped
            ),
            lead_clamped=actuation.lead_clamped,
            range_clamped=actuation.range_clamped,
        )
        return trace, safety


class ReactivePositionPipeline:
    """Two independent arm pipelines behind one combined-command boundary."""

    def __init__(
        self,
        plant_state,
        setup,
        controller_config=CONTROL,
        human_safety_config=CONFIG.human_safety,
    ):
        if not isinstance(plant_state, PlantState):
            raise TypeError("plant_state must be a PlantState")
        if not isinstance(setup, DualArmPipelineSetup):
            raise TypeError("setup must be a DualArmPipelineSetup")
        if not isinstance(controller_config, ReactivePoseConfig):
            raise TypeError("controller_config must be ReactivePoseConfig")
        if not isinstance(human_safety_config, HumanSafetyConfig):
            raise TypeError(
                "human_safety_config must be HumanSafetyConfig"
            )

        self._last_human_safety_statuses = {}
        max_human_constraints = sum(
            not sphere.mount_exempt for sphere in LINK_SPHERES
        )
        self._arms = {
            side: ArmControlPipeline(
                plant_state.arm(side).position_rad,
                setup.for_arm(side),
                controller_config,
                human_safety_config,
                max_human_constraints,
            )
            for side in ARMS
        }

    def command(self):
        """Return the current persistent position command for both arms."""
        return JointPositionCommand(
            right_position_rad=self._arms["right"].command_rad,
            left_position_rad=self._arms["left"].command_rad,
        )

    @property
    def human_safety_statuses(self):
        return dict(self._last_human_safety_statuses)

    def step(
        self,
        states,
        targets,
        dt_s,
        arms=ARMS,
        human_safety_states=None,
    ):
        """Compute one command without reading or writing a plant backend."""
        if not isinstance(states, DualArmControllerStates):
            raise TypeError("states must be DualArmControllerStates")
        if not isinstance(targets, DualArmWorldTargets):
            raise TypeError("targets must be DualArmWorldTargets")
        if (
            human_safety_states is not None
            and not isinstance(
                human_safety_states, DualArmHumanSafetyStates
            )
        ):
            raise TypeError(
                "human_safety_states must be DualArmHumanSafetyStates "
                "or None"
            )
        selected = tuple(arms)
        if len(set(selected)) != len(selected):
            raise ValueError("arms must not contain duplicates")
        for side in selected:
            if side not in ARMS:
                raise ValueError(f"unknown arm: {side!r}")

        traces = {"right": None, "left": None}
        safety_statuses = {}
        for side in selected:
            traces[side], safety_statuses[side] = self._arms[side].step(
                states.for_arm(side),
                targets.for_arm(side),
                dt_s,
                None
                if human_safety_states is None
                else human_safety_states.for_arm(side),
            )
        self._last_human_safety_statuses = safety_statuses
        return self.command(), DualArmControlTraces(**traces)
