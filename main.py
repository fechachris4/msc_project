import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos, pd
from sim import world

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

desired_pos.apply()

step = 0
with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(world.model, world.data)

        if step % PRINT_EVERY == 0:
            for name, pose_error in (("right", pd.right_pose_error),
                                     ("left", pd.left_pose_error)):
                e_pos, _ = pose_error()
                print(f"t={world.data.time:6.2f}s  {name:5s} "
                      f"|e|={np.linalg.norm(e_pos):.3f} m  "
                      f"e_pos=[{e_pos[0]: .3f} {e_pos[1]: .3f} {e_pos[2]: .3f}]")
        step += 1

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
