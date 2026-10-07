# Holding a robot's hand still while its wearer moves

[![tests](https://github.com/fechachris4/msc_project/actions/workflows/tests.yml/badge.svg)](https://github.com/fechachris4/msc_project/actions/workflows/tests.yml)

Christian Akabueze · MSc Human and Biological Robotics, Imperial College London (MUVE Lab) · 2026

Two Kinova Gen3 arms are worn on the torso as extra limbs. When the wearer walks, the mount that carries the arms bounces and sways, and anything the arms hold moves with it. My MSc project asked how much of that motion the arms can cancel, so their end-effectors (the robot's hands) stay fixed in the room rather than on the body.

![The rig: a participant walking on the treadmill with both arms on, and the same rig from the front](media/rig.jpg)

*The rig. Left: a participant walking on the treadmill with both arms on. Right: the same rig from the front. Vicon cameras on the truss and on tripods track the mount and the end-effectors.*

**On hardware**, seven participants walked on a treadmill while the arms held a fixed point in the room. Averaged over six of them (the first session is reported separately), the arms removed 71% of the mount motion at a slow walk (0.5 m/s), 63% at a normal walk (1.0 m/s) and 56% at a fast walk (1.5 m/s). Slow sway was mostly removed. The faster bounce that comes with each step largely was not.

**The likely reason is timing.** The arm learns how the mount is moving 61-71 ms late. That is the delay I measured on the rig, and a model with that delay matches the measured response.

**In simulation**, where that signal is exact, using it to cancel the mount's motion as it happens cuts the error from 7.5 mm to 2.6 mm at 1.8 Hz, about the rate of steps in walking. Delay the signal by 60 ms and the error is back to 8.2-8.6 mm. So the next thing to improve on hardware is the velocity estimate, ahead of the control law.

<p align="center"><img src="media/hw_turn.gif" width="480" alt="The wearer twists his torso; the end-effectors move much less"></p>

*Phone video from the lab, stabilised on the background. The controller is in world hold during setup (not a recorded trial). The wearer twists his torso by about 60°; the elbows swing, while the end-effectors move much less than the torso.*

This repo is the MuJoCo simulation and controller I built for the project ("World-Stable Supernumerary Effectors for Human Augmentation Under Locomotion-Scale Base Motion"), plus the scripts behind the hardware figures. To run it, see [Run the simulation](#run-the-simulation). There is also a C++20 port in [`cpp/`](cpp/README.md). The arm model is from MuJoCo Menagerie; the controller, safety filter, experiments and C++ port are mine. The hardware numbers here match the thesis, which is under examination; I will link it once it is marked.

## Results in pictures

One hardware trial at 1.0 m/s, replayed from the motion-capture data. Grey is where the end-effector would have gone if the arm were locked to the mount. Blue is where it went.

![One trial replayed from the Vicon data](media/hw_replay.gif)

The same comparison in simulation, at 1.8 Hz: arms locked, the reactive controller, and the reactive controller with feedforward added.

![Arms locked vs reactive vs reactive + feedforward](media/hold_pose.gif)

The detail is on two pages:

- [Hardware trials](docs/hardware-trials.md): how a trial is scored, results by speed and frequency, and what limits the arm.
- [Simulation](docs/simulation.md): the delay experiment, how the controller works, and the frequency sweep.

## Scope and next steps

- Hardware results come from seven participants (six in the pooled means), mostly one arm at a time. Next: both arms together, randomised speed order, and a trial that changes the velocity estimator.
- Feedforward is validated in simulation; running it on hardware is the natural next test.
- The simulated disturbance is a controlled sum of sinusoids. Replaying recorded gait from the Vicon data is the next step.
- The simulated delay is a pure delay; adding the estimator's noise and filtering would match hardware more closely.
- Joint limits are handled by clamping; joint-limit avoidance would extend the workspace for starts far from the target.
- The simulated arms are position-servoed and rigid, as modelled in MuJoCo.
- This repo establishes the reactive baseline plus feedforward. A predictive controller (MPC) that compensates the measured delay is the planned comparison.

## Code

| Path | |
|---|---|
| `controller/reactive_controller.py` | The control law in order: errors, PD, feedforward, DLS, null space, safety projection |
| `controller/frames.py`, `pin_fk.py` | World-frame kinematics composed through the mount (Pinocchio FK) |
| `controller/human_safety.py`, `link_spheres.py` | Arm spheres, the wearer cylinder and the distance constraints |
| `controller/runner.py`, `servo.py` | One control cycle; MuJoCo and a future Kinova hardware backend share one interface ([docs/backend-contract.md](docs/backend-contract.md)) |
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
| `analysis/` | FK against MuJoCo, end-effector velocity ([docs/velocity-validation.md](docs/velocity-validation.md)), tracking bandwidth, live dashboard |
| `tests/` | 261 unit and closed-loop tests |

Also in the repo, not used for the results: Cartesian target trajectories, including look-at orientation targets ([docs/cartesian-trajectories.md](docs/cartesian-trajectories.md)) and a collision-aware path planner above the controller (`planning/`, [docs/planning.md](docs/planning.md)).

Two FK implementations exist on purpose: `pin_fk.py` (Pinocchio) is the control path, `kinematics.py` (analytical, from MuJoCo model constants) cross-checks it. Frames are written `T_A_B`: pose of frame B in frame A. Everything is SI internally; millimetres only in prints and plots. In the code the mount is also called the torso (`torso_pose_world`).

## Run the simulation

Tested on Python 3.14.4 with `mujoco`, `pin`, `numpy`, `scipy`, `osqp`, `matplotlib`, and `pandas` for the hardware figure scripts (`requirements.txt`; pinned versions in `requirements-lock.txt`). Run from the repo root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mjpython tools/mount_disturbance.py both   # viewer at f = 1.8 Hz (mjpython on macOS)
python tools/velocity_delay.py             # delay figure, about 4 minutes
python tools/disturbance_freq_sweep.py     # results table and sweep figure
python tools/feedforward_compare.py right --f=1.8
python tools/make_readme_media.py          # the simulation video
python -m unittest discover tests
```

Gains, limits, targets and the safety envelope are in `config/control.toml`. The Kinova Gen3 model in `sim/assets/kinova_gen3/` is from MuJoCo Menagerie (BSD licence, Kinova). My code is MIT licensed.

## Contact

For questions about the project or the code, [open an issue](https://github.com/fechachris4/msc_project/issues) or email fecha412@gmail.com. The repository is maintained by Christian Akabueze.
