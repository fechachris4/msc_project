"""FK validation: compare the direct (MuJoCo-measured) right EE position
with the FK-composed position from controller.kinematics.

Simulates headless for DURATION seconds, then plots direct vs FK per axis
and saves the figure to plots/. No viewer, so plain python works:

    python -m plotting.fk_validation
"""

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller import kinematics
from sim import world

DURATION = 5.0  # seconds of simulation


def simulate():
    n_steps = int(DURATION / world.model.opt.timestep)
    times = np.empty(n_steps)
    direct = np.empty((n_steps, 3))
    fk = np.empty((n_steps, 3))

    for i in range(n_steps):
        mujoco.mj_step(world.model, world.data)
        times[i] = world.data.time
        direct[i] = world.data.site_xpos[world.right_ee_id]
        fk[i], _ = kinematics.right_ee_positions()

    return times, direct, fk


def plot(times, direct, fk):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for axis, (ax, label) in enumerate(zip(axes, "XYZ")):
        ax.plot(times, direct[:, axis], label="direct (MuJoCo)")
        ax.plot(times, fk[:, axis], "--", label="FK")
        ax.set_ylabel(f"{label} (m)")
        ax.legend()
    axes[-1].set_xlabel("sim time (s)")
    fig.suptitle("Right EE position: direct vs FK")

    max_error = np.abs(direct - fk).max()
    print(f"max |direct - FK| error: {max_error:.2e} m")

    fig.savefig("plots/fk_validation.png", dpi=150)
    print("Saved plots/fk_validation.png")
    plt.show()


if __name__ == "__main__":
    plot(*simulate())
