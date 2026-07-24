"""Explicit runner for the current reactive pose-to-position pipeline."""

from dataclasses import dataclass

from controller import frames
from controller.backend import PlantBackend
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
)
from controller.trajectory import (
    as_dual_arm_target_source,
    sample_dual_arm_target_source,
)
from runtime_config import CONFIG, ReactivePoseConfig


@dataclass(frozen=True, slots=True)
class RunnerCycle:
    """One completed exchange, retaining both sides of the boundary."""

    input_state: PlantState
    target_elapsed_time_s: float
    sampled_targets: DualArmFramedTargets
    resolved_targets: DualArmWorldTargets
    controller_states: DualArmControllerStates
    command: JointPositionCommand
    traces: DualArmControlTraces
    next_state: PlantState


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
        command, traces = self._pipeline.step(
            controller_states,
            resolved,
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
            controller_states=controller_states,
            command=command,
            traces=traces,
            next_state=next_state,
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
