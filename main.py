"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  target mocap pose (world) + joint angles qpos  [sim/targets, MuJoCo]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose error e = ref - actual (world)           [controller/servo]
  -> commanded twist v = Kp*e -> qdot via DLS      [controller/servo]
  -> integrate position-servo setpoints data.ctrl (rad)  [controller/servo]
  -> mj_step
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos, servo
from sim import world

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

choice = sys.argv[1] if len(sys.argv) > 1 else "both"
assert choice in ("right", "left", "both"), "usage: python main.py [right|left|both]"
arms = ("right", "left") if choice == "both" else (choice,)

desired_pos.apply()
servo.init_ctrl()

step = 0
with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        servo.apply_ctrl(world.model.opt.timestep, arms)
        mujoco.mj_step(world.model, world.data)

        if step % PRINT_EVERY == 0:
            for side in world.SIDES:
                e_pos, _ = servo.pose_error(side)
                e_mm = e_pos * 1000.0
                print(f"t={world.data.time:6.2f}s  {side:5s} "
                      f"|e|={np.linalg.norm(e_mm):.1f} mm  "
                      f"e_pos=[{e_mm[0]: 7.1f} {e_mm[1]: 7.1f} {e_mm[2]: 7.1f}]")
        step += 1

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
