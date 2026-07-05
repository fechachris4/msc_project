import time
from collections import deque

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

from sim import world
from pd import controller_step
from controller import kinematics

BUFFER_SIZE = 2000
PLOT_EVERY_N_STEPS = 25

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("sim time (s)")
ax.set_ylabel("position (m)")

times = deque(maxlen=BUFFER_SIZE)
series = {
    "MuJoCo X": (deque(maxlen=BUFFER_SIZE), "tab:blue"),
    "FK X": (deque(maxlen=BUFFER_SIZE), "tab:orange"),
    "MuJoCo Y": (deque(maxlen=BUFFER_SIZE), "tab:green"),
    "FK Y": (deque(maxlen=BUFFER_SIZE), "tab:red"),
    "MuJoCo Z": (deque(maxlen=BUFFER_SIZE), "tab:purple"),
    "FK Z": (deque(maxlen=BUFFER_SIZE), "tab:brown"),
}
lines = {
    name: ax.plot([], [], color=color, label=name)[0]
    for name, (_, color) in series.items()
}
ax.legend(loc="upper right")

with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
    step_count = 0
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(world.model, world.data)

        mujoco_pos = world.data.site_xpos[world.right_ee_id]
        fk_pos, _ = kinematics.right_ee_positions()

        times.append(world.data.time)
        for axis, name_pair in enumerate(
            [("MuJoCo X", "FK X"), ("MuJoCo Y", "FK Y"), ("MuJoCo Z", "FK Z")]
        ):
            series[name_pair[0]][0].append(mujoco_pos[axis])
            series[name_pair[1]][0].append(fk_pos[axis])

        step_count += 1
        if step_count % PLOT_EVERY_N_STEPS == 0:
            for name, (buffer, _) in series.items():
                lines[name].set_data(times, buffer)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        viewer.sync()

        time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
