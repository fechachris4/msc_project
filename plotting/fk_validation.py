"""Live FK validation: compare the direct (MuJoCo-measured) right EE position
with the FK-composed position from controller.kinematics, plotted while the
simulation runs (headless — no MuJoCo viewer, so plain python works):

    python -m plotting.fk_validation

Close the plot window to stop; the figure is saved to plots/fk_validation.png.
"""

from collections import deque

import matplotlib.pyplot as plt
import mujoco

from controller import kinematics
from sim import world

WINDOW = 2000  # samples kept on screen
REDRAW_EVERY = 25  # sim steps per plot refresh


def main():
    plt.ion()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    buffers = [deque(maxlen=WINDOW) for _ in range(7)]  # t, direct xyz, fk xyz
    lines = []
    for axis, (ax, label) in enumerate(zip(axes, "XYZ")):
        lines.append(ax.plot([], [], label="direct (MuJoCo)")[0])
        lines.append(ax.plot([], [], "--", label="FK")[0])
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("sim time (s)")
    fig.suptitle("Right EE position: direct vs FK")

    step = 0
    while plt.fignum_exists(fig.number):
        mujoco.mj_step(world.model, world.data)
        fk_pos, _ = kinematics.right_ee_positions()
        direct_pos = world.data.site_xpos[world.right_ee_id]

        buffers[0].append(world.data.time)
        for axis in range(3):
            buffers[1 + 2 * axis].append(direct_pos[axis])
            buffers[2 + 2 * axis].append(fk_pos[axis])

        step += 1
        if step % REDRAW_EVERY == 0:
            for line, buffer in zip(lines, buffers[1:]):
                line.set_data(buffers[0], buffer)
            for ax in axes:
                ax.relim()
                ax.autoscale_view()
            plt.pause(0.001)  # redraw and let the GUI breathe

    fig.savefig("plots/fk_validation.png", dpi=150)
    print("Saved plots/fk_validation.png")


if __name__ == "__main__":
    main()
