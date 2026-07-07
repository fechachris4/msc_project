"""Live position error: displacement of one arm's FK end-effector position
from its fixed world-frame target, plotted while the simulation runs
(headless — no MuJoCo viewer, so plain python works). Shared implementation;
run via the per-arm entry points:

    python -m plotting.right.position_error
    python -m plotting.left.position_error

Close the plot window to stop; the figure is saved to
plots/<arm>_position_error.png. Display is millimetres (math stays metres).
"""

import mujoco
import numpy as np

from controller import desired_pos
from plotting.live_plot import LivePlot
from sim import world


def run(arm_name, pose_error):
    """arm_name: "right" or "left"; pose_error: pd.<arm>_pose_error."""
    desired_pos.apply()

    plot = LivePlot(
        rows=["e_x (mm)", "e_y (mm)", "e_z (mm)", "|e| (mm)"],
        signals={"error": {}},
        title=f"{arm_name} EE position error (world frame)",
    )
    while plot.is_open():
        mujoco.mj_step(world.model, world.data)
        # refresh poses: mj_step advances qpos after computing them
        mujoco.mj_kinematics(world.model, world.data)
        e_pos, _ = pose_error()
        e_mm = e_pos * 1000.0
        plot.add(world.data.time,
                 {"error": [e_mm[0], e_mm[1], e_mm[2], np.linalg.norm(e_mm)]})

    path = f"plots/{arm_name}_position_error.png"
    plot.save(path)
    print(f"Saved {path}")
