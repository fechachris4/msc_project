import time

import mujoco
import mujoco.viewer

from controller import reference
from sim import world

reference.apply()

with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(world.model, world.data)

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
