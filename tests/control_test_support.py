"""Small explicit MuJoCo harness used by control-loop regression tests."""

from controller import frames, reactive_pose, servo
from controller.state import Twist
from sim import targets, world


def reconstruct_pipeline(control=servo.CONTROL):
    plant = world.read_state(Twist.zero())
    pipeline = servo.ReactivePositionPipeline(
        plant, world.PIPELINE_SETUP, control)
    world.apply_command(pipeline.command())
    return pipeline


def apply_cycle(
    pipeline,
    dt_s,
    base_twist,
    arms=world.SIDES,
    world_targets=None,
):
    plant = world.read_state(Twist(*base_twist))
    command, traces = pipeline.step(
        frames.controller_states(plant, world.MOUNT_CALIBRATION),
        targets.world_targets() if world_targets is None else world_targets,
        dt_s,
        arms,
    )
    world.apply_command(command)
    return traces


def pose_error(side):
    plant = world.read_state(Twist.zero())
    state = frames.arm_controller_state(
        plant, side, world.MOUNT_CALIBRATION)
    return reactive_pose.pose_error(state, targets.world_target(side))
