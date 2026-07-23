"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  target mocap pose (world) + joint angles qpos  [sim/targets, MuJoCo]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors e, e_v (world)            [controller/servo]
  -> commanded twist v = Kp*e + Kd*e_v -> qdot via DLS  [controller/servo]
  -> integrate position-servo setpoints data.ctrl (rad)  [controller/servo]
  -> mj_step

usage: mjpython main.py [right|left|both]
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos, frames, servo
from controller.state import Twist
from runtime_config import CONFIG, print_effective_config
from sim import motion, world

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

def _parse_args(argv):
    values = list(argv)
    usage = "usage: mjpython main.py [right|left|both]"
    if len(values) > 1:
        raise SystemExit(usage)
    choice = values[0] if values else "both"
    if choice not in ("right", "left", "both"):
        raise SystemExit(usage)
    return world.SIDES if choice == "both" else (choice,)


def main(argv=None):
    import sys

    arms = _parse_args(sys.argv[1:] if argv is None else argv)
    print_effective_config(CONFIG)
    source_targets = desired_pos.apply()
    servo.init_ctrl()

    step = 0
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            motion.set_torso_pose(world.data.time)
            # Refresh xpos/xmat from the mocap write: without this the
            # controller sees the torso pose of the previous step (t - dt)
            # paired with the base twist at t.
            mujoco.mj_kinematics(world.model, world.data)
            # set_torso_pose (mocap write) and torso_twist_at (feedforward)
            # must stay a matched pair — same scenario, same instant t.
            base_twist = motion.torso_twist_at(world.data.time)
            plant = world.read_state(Twist(*base_twist))
            world_targets = frames.resolve_targets_world(
                plant, world.MOUNT_CALIBRATION, source_targets)
            desired_pos.show_targets(world_targets)
            servo.apply_ctrl(
                world.model.opt.timestep,
                base_twist,
                arms,
                world_targets=world_targets,
            )
            mujoco.mj_step(world.model, world.data)

            if step % PRINT_EVERY == 0:
                for side in world.SIDES:
                    e_pos, _ = servo.pose_error(side)
                    e_mm = e_pos * 1000.0
                    # sigma_min -> 0 means a task direction is being lost.
                    state = frames.arm_controller_state(
                        world.read_state(Twist.zero()),
                        side,
                        world.MOUNT_CALIBRATION,
                    )
                    sigma = np.linalg.svd(
                        state.jacobian_world, compute_uv=False)
                    print(f"t={world.data.time:6.2f}s  {side:5s} "
                          f"|e|={np.linalg.norm(e_mm):.1f} mm  "
                          f"e_pos=[{e_mm[0]: 7.1f} {e_mm[1]: 7.1f} "
                          f"{e_mm[2]: 7.1f}]  "
                          f"sigma=[{' '.join(f'{s:.3f}' for s in sigma)}]")
            step += 1

            viewer.sync()

            time_until_next_step = (
                world.model.opt.timestep - (time.perf_counter() - step_start))
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
