"""Live plot comparing the direct (MuJoCo-measured) right EE position
against the FK-composed position from controller.kinematics.

Run standalone from the repo root (macOS needs mjpython for the viewer):

    mjpython -m plotting.fk_validation

The matplotlib window runs in a separate process: mjpython keeps the OS main
thread for the MuJoCo viewer, and the macOS matplotlib backend also requires
a main thread, so the two cannot share one process.

On exit the figure is saved to plots/fk_validation_<timestamp>.png.
"""

import multiprocessing
import queue as queue_module
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

BUFFER_SIZE = 2000
PLOT_EVERY_N_SAMPLES = 25
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


class FKValidationPlot:
    """Live figure with direct vs FK right-EE position, one pair of lines per axis."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
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
        self.sample_count = 0

    def record(self, sim_time, direct_pos, fk_pos):
        self.times.append(sim_time)
        for axis, (direct_name, fk_name) in enumerate(
            [("MuJoCo X", "FK X"), ("MuJoCo Y", "FK Y"), ("MuJoCo Z", "FK Z")]
        ):
            self.series[direct_name][0].append(direct_pos[axis])
            self.series[fk_name][0].append(fk_pos[axis])

        self.sample_count += 1
        if self.sample_count % PLOT_EVERY_N_SAMPLES == 0:
            self.redraw()

    def redraw(self):
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


def plotter_process(sample_queue):
    """Runs in its own process so matplotlib gets its own main thread.

    Consumes (sim_time, direct_pos, fk_pos) tuples; a None sentinel ends the
    run and saves the figure.
    """
    plot = FKValidationPlot()
    running = True
    while running:
        try:
            item = sample_queue.get(timeout=0.05)
        except queue_module.Empty:
            plot.plt.pause(0.01)  # keep the window responsive while idle
            continue
        if item is None:
            running = False
        else:
            plot.record(*item)

    plot.redraw()
    path = plot.save()
    print(f"Saved plot to {path}")


def start_plotter():
    """Spawn the plotting process; returns (process, queue)."""
    ctx = multiprocessing.get_context("spawn")
    # Under mjpython, respawning via the mjpython binary would grab another
    # Cocoa main loop; spawn the child with the plain interpreter instead.
    exe = Path(sys.executable)
    if exe.name == "mjpython":
        for candidate in (exe.parent / "python3", exe.parent / "python"):
            if candidate.exists():
                multiprocessing.set_executable(str(candidate))
                break
    sample_queue = ctx.Queue()
    process = ctx.Process(target=plotter_process, args=(sample_queue,), daemon=True)
    process.start()
    return process, sample_queue


def main():
    import mujoco
    import mujoco.viewer

    from controller import kinematics
    from sim import world

    process, sample_queue = start_plotter()

    try:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(world.model, world.data)

                direct_pos = world.data.site_xpos[world.right_ee_id].copy()
                fk_pos, _ = kinematics.right_ee_positions()
                sample_queue.put((world.data.time, direct_pos, fk_pos))

                viewer.sync()

                time_until_next_step = world.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        sample_queue.put(None)
        process.join(timeout=10)


if __name__ == "__main__":
    main()
