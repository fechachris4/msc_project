"""Live plot comparing the direct (MuJoCo-measured) right EE position
against the FK-composed position from controller.kinematics.

Run standalone from the repo root (macOS needs mjpython for the viewer):

    mjpython -m plotting.fk_validation

Plots appear in the Rerun viewer, which opens as its own app — no matplotlib
main-thread conflicts with mjpython. Use the viewer's "Save" to keep a run.
"""

import time

import mujoco
import mujoco.viewer
import rerun as rr

from controller import kinematics
from sim import world

AXES = ("x", "y", "z")


def main():
    rr.init("fk_validation", spawn=True)

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(world.model, world.data)

            direct_pos = world.data.site_xpos[world.right_ee_id]
            fk_pos, _ = kinematics.right_ee_positions()

            rr.set_time("sim_time", duration=world.data.time)
            for axis, name in enumerate(AXES):
                rr.log(f"right_ee/{name}/mujoco", rr.Scalars(direct_pos[axis]))
                rr.log(f"right_ee/{name}/fk", rr.Scalars(fk_pos[axis]))

            viewer.sync()

            time_until_next_step = world.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
