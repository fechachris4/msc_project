# Code map

Where things live in the repo.

| Path | |
|---|---|
| `controller/reactive_controller.py` | The control law in order: errors, PD, feedforward, DLS, null space, safety projection |
| `controller/frames.py`, `pin_fk.py` | World-frame kinematics composed through the mount (Pinocchio FK) |
| `controller/human_safety.py`, `link_spheres.py` | Arm spheres, the wearer cylinder and the distance constraints |
| `controller/runner.py`, `servo.py` | One control cycle; MuJoCo and a future Kinova hardware backend share one interface ([docs/backend-contract.md](backend-contract.md)) |
| `sim/` | MJCF scene, MuJoCo backend |
| `tools/mount_disturbance.py` | The disturbance model, and a viewer that runs it |
| `tools/velocity_delay.py` | Error against delay on the mount-velocity signal |
| `tools/disturbance_freq_sweep.py` | Frequency sweep (results table and figure) |
| `tools/feedforward_compare.py` | Reactive vs feedforward, error over time |
| `tools/gain_sweep.py` | Kp × Kd sweep behind the chosen position gains |
| `tools/disturbance_run.py` | Headless disturbance run: report figure, GIF and CSV |
| `tools/velocity_ff_demo.py` | Trajectory-velocity feedforward on a moving target, static mount |
| `main.py`, `arm_flow.py` | Viewer entry point; per-arm reference composition (goal or trajectory, optional planner) |
| `runtime_config.py` | Strict loader for the control TOML shared with the C++ port |
| `tools/make_readme_media.py` | The simulation video |
| `tools/hardware_figures.py`, `hardware_replay.py` | The hardware figures and replay (need the thesis data, not included) |
| `analysis/` | FK against MuJoCo, end-effector velocity ([docs/velocity-validation.md](velocity-validation.md)), tracking bandwidth, live dashboard |
| `tests/` | 261 unit and closed-loop tests |

Also in the repo, not used for the results: Cartesian target trajectories, including look-at orientation targets ([docs/cartesian-trajectories.md](cartesian-trajectories.md)) and a collision-aware path planner above the controller (`planning/`, [docs/planning.md](planning.md)).

Two FK implementations exist on purpose: `pin_fk.py` (Pinocchio) is the control path, `kinematics.py` (analytical, from MuJoCo model constants) cross-checks it. Frames are written `T_A_B`: pose of frame B in frame A. Everything is SI internally; millimetres only in prints and plots. In the code the mount is also called the torso (`torso_pose_world`).

## Setup details

Tested on Python 3.14.4 with `mujoco`, `pin`, `numpy`, `scipy`, `osqp`, `matplotlib`, and `pandas` for the hardware figure scripts (`requirements.txt`; pinned versions in `requirements-lock.txt`). Gains, limits, targets and the safety envelope are in `config/control.toml`. The Kinova Gen3 model is in `sim/assets/kinova_gen3/`.

## Regenerating the figures

Run from the repo root, after the setup in the [README](../README.md#run-the-simulation).

```bash
python tools/velocity_delay.py             # delay figure, about 4 minutes
python tools/disturbance_freq_sweep.py     # results table and sweep figure
python tools/feedforward_compare.py right --f=1.8
python tools/make_readme_media.py          # the simulation video
```
