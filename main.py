"""Viewer entry point: world-frame EE pose hold, closed loop.

Per-step data flow (all SI: meters, radians; mm only in the printout):
  target mocap pose (world) + joint angles qpos  [sim/targets, MuJoCo]
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  [controller/frames]
  -> pose + twist errors e, e_v (world)            [controller/servo]
  -> commanded twist v = Kp*e + Kd*e_v -> qdot via DLS  [controller/servo]
  -> integrate position-servo setpoints data.ctrl (rad)  [controller/servo]
  -> backend.exchange: apply command, mj_step, return next state

usage: mjpython main.py [right|left|both]
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import desired_pos
from controller.runner import ReactivePositionRunner
from runtime_config import CONFIG, print_effective_config
from sim import motion, world

PRINT_EVERY = 250  # steps between error printouts (0.5 s at the 2 ms timestep)

def _parse_args(argv):
    values = list(argv)
    usage = "usage: mjpython main.py [right|left|both]"
    if len(values) > 1:
        raise SystemExit(usage)
    choice = values[0] if values else "both"
    if choice not in ("right", "left", "both"):
        raise SystemExit(usage)
    return world.SIDES if choice == "both" else (choice,)


def main(argv=None):
    import sys

    arms = _parse_args(sys.argv[1:] if argv is None else argv)
    print_effective_config(CONFIG)
    source_targets = desired_pos.apply()
    world.backend.configure_torso_driver(
        motion.torso_pose_at, motion.torso_twist_at)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        source_targets,
        arms,
    )
    runner.start()

    step = 0
    try:
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                cycle = runner.cycle()
                desired_pos.show_targets(cycle.resolved_targets)

                if step % PRINT_EVERY == 0:
                    for side, trace in cycle.traces.items():
                        e_pos = trace.e_pos
                        state = cycle.input_state
                        sample_time = state.sample_time_s
                        e_mm = e_pos * 1000.0
                        sigma = np.linalg.svd(
                            trace.J, compute_uv=False)
                        print(
                            f"t={sample_time:6.2f}s  {side:5s} "
                            f"|e|={np.linalg.norm(e_mm):.1f} mm  "
                            f"e_pos=[{e_mm[0]: 7.1f} {e_mm[1]: 7.1f} "
                            f"{e_mm[2]: 7.1f}]  "
                            f"sigma=[{' '.join(f'{s:.3f}' for s in sigma)}]"
                        )
                step += 1

                viewer.sync()

                time_until_next_step = (
                    world.model.opt.timestep
                    - (time.perf_counter() - step_start)
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)


if __name__ == "__main__":
    main()
