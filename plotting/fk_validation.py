"""Live FK validation: compare the direct (MuJoCo-measured) right EE position
with the FK-composed position from controller.kinematics, plotted while the
simulation runs (headless — no MuJoCo viewer, so plain python works):

    python -m plotting.fk_validation

Close the plot window to stop; the figure is saved to plots/fk_validation.png.
"""

import mujoco

from controller import kinematics
from plotting.live_plot import LivePlot
from sim import world


def sample():
    """One comparison sample: (sim time, direct EE xyz, FK-composed EE xyz)."""
    fk_pos, _ = kinematics.right_ee_positions()
    direct_pos = world.data.site_xpos[world.right_ee_id].copy()
    return world.data.time, direct_pos, fk_pos


def main():
    plot = LivePlot(
        rows=["X (m)", "Y (m)", "Z (m)"],
        signals={"direct (MuJoCo)": {}, "FK": {"style": "--"}},
        title="Right EE position: direct vs FK",
    )
    while plot.is_open():
        mujoco.mj_step(world.model, world.data)
        t, direct_pos, fk_pos = sample()
        plot.add(t, {"direct (MuJoCo)": direct_pos, "FK": fk_pos})

    plot.save("plots/fk_validation.png")
    print("Saved plots/fk_validation.png")


if __name__ == "__main__":
    main()
