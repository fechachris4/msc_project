"""Live FK validation: compare MuJoCo's measured right EE position with the
analytical FK (extract_chain/fk, computed from qpos and model constants only),
plotted while the simulation runs (headless — no MuJoCo viewer, so plain
python works):

    python -m plotting.fk_validation

Close the plot window to stop; the figure is saved to plots/fk_validation.png.
The two traces are computed independently — agreement here validates the
analytical forward kinematics, not just frame algebra.
"""

import mujoco

from controller import frames
from plotting.live_plot import LivePlot
from sim import world


def sample():
    """One comparison sample: (sim time, direct EE xyz, FK-composed EE xyz)."""
    fk_pos, _ = frames.right_ee_pose()
    direct_pos = world.data.site_xpos[world.right_ee_id].copy()
    return world.data.time, direct_pos, fk_pos


def main():
    plot = LivePlot(
        rows=["X (mm)", "Y (mm)", "Z (mm)"],
        signals={"direct (MuJoCo)": {}, "FK": {"style": "--"}},
        title="Right EE position: direct vs FK",
    )
    while plot.is_open():
        mujoco.mj_step(world.model, world.data)
        # refresh site poses: mj_step advances qpos after computing them
        mujoco.mj_kinematics(world.model, world.data)
        t, direct_pos, fk_pos = sample()
        plot.add(t, {"direct (MuJoCo)": direct_pos * 1000.0,
                     "FK": fk_pos * 1000.0})

    plot.save("plots/fk_validation.png")
    print("Saved plots/fk_validation.png")


if __name__ == "__main__":
    main()
