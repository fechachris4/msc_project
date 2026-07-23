"""Live FK validation: compare MuJoCo's measured right EE position with
frames.ee_pose (the Pinocchio-backed T_W_T · T_T_K · T_K_E composition the
controller actually uses), plotted while the simulation runs (headless —
no MuJoCo viewer, so plain python works):

    python -m analysis.fk_validation

Close the plot window to stop; the figure is saved to
analysis/output/fk_validation.png. The two traces are computed
independently — agreement here validates the analytical forward
kinematics, not just frame algebra.
"""

import mujoco

from controller import frames
from controller.state import Twist
from plotting.live_plot import LivePlot
from sim import world


def sample():
    """One comparison sample: (sim time, direct EE xyz, FK-composed EE xyz)."""
    state = frames.arm_controller_state(
        world.read_state(Twist.zero()),
        "right",
        world.MOUNT_CALIBRATION,
    )
    fk_pos = state.ee_pose_world.position_m
    direct_pos = world.measured_ee_pose("right").position_m
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

    plot.save("analysis/output/fk_validation.png")
    print("Saved analysis/output/fk_validation.png")


if __name__ == "__main__":
    main()
