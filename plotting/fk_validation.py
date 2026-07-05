"""Live plot comparing the direct (MuJoCo-measured) right EE position
against the FK-composed position from controller.kinematics.

Run standalone from the repo root:

    python -m plotting.fk_validation

On exit the figure is saved to plots/fk_validation_<timestamp>.png.
"""

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

from controller import kinematics
from sim import world

BUFFER_SIZE = 2000
PLOT_EVERY_N_STEPS = 25
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


class FKValidationPlot:
    """Live figure with direct vs FK right-EE position, one pair of lines per axis."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("sim time (s)")
        self.ax.set_ylabel("position (m)")

        self.times = deque(maxlen=BUFFER_SIZE)
        self.series = {
            "MuJoCo X": (deque(maxlen=BUFFER_SIZE), "tab:blue"),
            "FK X": (deque(maxlen=BUFFER_SIZE), "tab:orange"),
            "MuJoCo Y": (deque(maxlen=BUFFER_SIZE), "tab:green"),
            "FK Y": (deque(maxlen=BUFFER_SIZE), "tab:red"),
            "MuJoCo Z": (deque(maxlen=BUFFER_SIZE), "tab:purple"),
            "FK Z": (deque(maxlen=BUFFER_SIZE), "tab:brown"),
        }
        self.lines = {
            name: self.ax.plot([], [], color=color, label=name)[0]
            for name, (_, color) in self.series.items()
        }
        self.ax.legend(loc="upper right")
        self.step_count = 0

    def record(self):
        """Sample direct and FK EE positions at the current sim state."""
        mujoco_pos = world.data.site_xpos[world.right_ee_id]
        fk_pos, _ = kinematics.right_ee_positions()

        self.times.append(world.data.time)
        for axis, (direct_name, fk_name) in enumerate(
            [("MuJoCo X", "FK X"), ("MuJoCo Y", "FK Y"), ("MuJoCo Z", "FK Z")]
        ):
            self.series[direct_name][0].append(mujoco_pos[axis])
            self.series[fk_name][0].append(fk_pos[axis])

        self.step_count += 1
        if self.step_count % PLOT_EVERY_N_STEPS == 0:
            self._redraw()

    def _redraw(self):
        for name, (buffer, _) in self.series.items():
            self.lines[name].set_data(self.times, buffer)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self):
        PLOTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PLOTS_DIR / f"fk_validation_{stamp}.png"
        self.fig.savefig(path, dpi=150)
        return path


def main():
    plot = FKValidationPlot()

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(world.model, world.data)

            plot.record()
            viewer.sync()

            time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    saved = plot.save()
    print(f"Saved plot to {saved}")


if __name__ == "__main__":
    main()
