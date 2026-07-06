import time

import mujoco
import mujoco.viewer

from sim import targets, world

# Start the targets at the current EE positions; overwrite with
# targets.set_right_target([x, y, z]) / set_left_target (world frame, m).
targets.reset_to_ee()

with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(world.model, world.data)

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
