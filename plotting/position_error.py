"""Live displacement from target: signed offset of one arm's FK end-effector
position from its fixed world-frame target, per axis, plotted while the
simulation runs (headless — no MuJoCo viewer, so plain python works):

    python -m plotting.position_error right
    python -m plotting.position_error left

Base motion follows this module's BASE_SCENARIO (zero amplitude = static).

Displayed as d = p_EE - p_target (world frame, mm): positive means the EE is
on the +axis side of the target. The dashed zero line is the target. Note the
sign is flipped from servo's control error (e = target - actual); the flip lives
here in the display layer only. Close the plot window to stop; the figure is
saved to plots/<side>_position_error.png.
"""

import sys

import mujoco
import numpy as np

from controller import desired_pos, servo
from plotting.live_plot import LivePlot
from sim import motion, world

# Base-motion scenario for this run — same values main.py runs today.
# Zero amplitudes = static base (hand-draggable torso).
BASE_SCENARIO = dict(
    linear_amplitude=np.array([0.1, 0.3, 0.0]),   # m, world xyz
    linear_frequency=0.1,                          # Hz
    rotational_amplitude=np.zeros(3),              # rad, rpy
    rotational_frequency=0.2,                      # Hz
)


def run(side):
    """side: "right" or "left".

    Closed loop: the PD controller runs every step, so the plot shows the
    controlled response (same loop body as main.py via servo.apply_ctrl)."""
    desired_pos.apply()
    servo.init_ctrl()

    plot = LivePlot(
        rows=["d_x (mm)", "d_y (mm)", "d_z (mm)", "|d| (mm)"],
        signals={"EE - target": {}, "target": {"style": "--"}},
        title=f"{side} EE displacement from target (world frame)",
    )
    while plot.is_open():
        motion.set_torso_pose(world.data.time, **BASE_SCENARIO)
        # refresh xpos/xmat so the controller sees the torso pose at t,
        # not the previous step's (same pattern as main.py)
        mujoco.mj_kinematics(world.model, world.data)
        # set_torso_pose (mocap write) and torso_twist_at (feedforward)
        # must stay a matched pair — same scenario, same instant t.
        servo.apply_ctrl(world.model.opt.timestep,
                         motion.torso_twist_at(world.data.time,
                                               **BASE_SCENARIO))
        mujoco.mj_step(world.model, world.data)
        # refresh poses: mj_step advances qpos after computing them
        mujoco.mj_kinematics(world.model, world.data)
        e_pos, _ = servo.pose_error(side)
        d_mm = -e_pos * 1000.0
        plot.add(world.data.time,
                 {"EE - target": [d_mm[0], d_mm[1], d_mm[2],
                                  np.linalg.norm(d_mm)],
                  "target": [0.0, 0.0, 0.0, 0.0]})

    path = f"plots/{side}_position_error.png"
    plot.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "right"
    assert side in world.SIDES, \
        "usage: python -m plotting.position_error [right|left]"
    run(side)
