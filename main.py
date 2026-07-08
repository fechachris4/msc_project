"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  target mocap pose (world) + joint angles qpos  [sim/targets, MuJoCo]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors e, e_v (world)            [controller/servo]
  -> commanded twist v = Kp*e + Kd*e_v -> qdot via DLS  [controller/servo]
  -> integrate position-servo setpoints data.ctrl (rad)  [controller/servo]
  -> mj_step
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos, frames, servo
from sim import motion, world

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

# Base-motion scenario for this run — the standing disturbance the
# reactive controller rejects. Zero amplitudes = static base
# (hand-draggable torso).
BASE_SCENARIO = dict(
    linear_amplitude=np.array([0.1, 0.3, 0.0]),   # m, world xyz
    linear_frequency=0.1,                          # Hz
    rotational_amplitude=np.zeros(3),              # rad, rpy
    rotational_frequency=0.2,                      # Hz
)

choice = sys.argv[1] if len(sys.argv) > 1 else "both"
assert choice in ("right", "left", "both"), \
    "usage: mjpython main.py [right|left|both]"
arms = world.SIDES if choice == "both" else (choice,)

desired_pos.apply()
servo.init_ctrl()

step = 0
with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        motion.set_torso_pose(world.data.time, **BASE_SCENARIO)
        # Refresh xpos/xmat from the mocap write: without this the
        # controller sees the torso pose of the previous step (t - dt)
        # paired with the base twist at t.
        mujoco.mj_kinematics(world.model, world.data)
        # set_torso_pose (mocap write) and torso_twist_at (feedforward)
        # must stay a matched pair — same scenario, same instant t.
        servo.apply_ctrl(world.model.opt.timestep,
                         motion.torso_twist_at(world.data.time,
                                               **BASE_SCENARIO), arms)
        mujoco.mj_step(world.model, world.data)

        if step % PRINT_EVERY == 0:
            for side in world.SIDES:
                e_pos, _ = servo.pose_error(side)
                e_mm = e_pos * 1000.0
                # Jacobian singular values, descending: sigma_min -> 0 means
                # a task direction is being lost (near-singular, DLS working).
                sigma = np.linalg.svd(frames.jacobian_world(side),
                                      compute_uv=False)
                print(f"t={world.data.time:6.2f}s  {side:5s} "
                      f"|e|={np.linalg.norm(e_mm):.1f} mm  "
                      f"e_pos=[{e_mm[0]: 7.1f} {e_mm[1]: 7.1f} {e_mm[2]: 7.1f}]  "
                      f"sigma=[{' '.join(f'{s:.3f}' for s in sigma)}]")
        step += 1

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
