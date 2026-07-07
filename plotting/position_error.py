"""Live displacement from target: signed offset of one arm's FK end-effector
position from its fixed world-frame target, per axis, plotted while the
simulation runs (headless — no MuJoCo viewer, so plain python works).
Shared implementation; run via the per-arm entry points:

    python -m plotting.right.position_error
    python -m plotting.left.position_error

Displayed as d = p_EE - p_target (world frame, mm): positive means the EE is
on the +axis side of the target. The dashed zero line is the target. Note the
sign is flipped from servo's control error (e = target - actual); the flip lives
here in the display layer only. Close the plot window to stop; the figure is
saved to plots/<arm>_position_error.png.
"""

import mujoco
import numpy as np

from controller import desired_pos, servo
from plotting.live_plot import LivePlot
from sim import world


def run(arm_name, pose_error):
    """arm_name: "right" or "left"; pose_error: servo.<arm>_pose_error.

    Closed loop: the P controller runs every step, so the plot shows the
    controlled response (same loop body as main.py via servo.apply_ctrl)."""
    desired_pos.apply()
    servo.init_ctrl()

    plot = LivePlot(
        rows=["d_x (mm)", "d_y (mm)", "d_z (mm)", "|d| (mm)"],
        signals={"EE - target": {}, "target": {"style": "--"}},
        title=f"{arm_name} EE displacement from target (world frame)",
    )
    while plot.is_open():
        servo.apply_ctrl(world.model.opt.timestep)
        mujoco.mj_step(world.model, world.data)
        # refresh poses: mj_step advances qpos after computing them
        mujoco.mj_kinematics(world.model, world.data)
        e_pos, _ = pose_error()
        d_mm = -e_pos * 1000.0
        plot.add(world.data.time,
                 {"EE - target": [d_mm[0], d_mm[1], d_mm[2],
                                  np.linalg.norm(d_mm)],
                  "target": [0.0, 0.0, 0.0, 0.0]})

    path = f"plots/{arm_name}_position_error.png"
    plot.save(path)
    print(f"Saved {path}")
