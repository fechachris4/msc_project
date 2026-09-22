# Holding a robot's hand still while its wearer moves

Two Kinova Gen3 arms are strapped to a person's torso as supernumerary robotic limbs. When the wearer walks, the mount bounces and sways, and anything the arms hold moves with it. I wanted to know how much of that motion a reactive controller can cancel, so the end-effectors stay fixed in the world rather than on the body.

This is the MuJoCo simulation and controller from my MSc project at Imperial (MUVE Lab), "World-Stable Supernumerary Effectors for Human Augmentation Under Locomotion-Based Motion".

![Arms locked vs reactive vs reactive + feedforward](media/hold_pose.gif)

*Same scripted mount motion in all three panels (f = 1.8 Hz, unscaled). Top: the whole robot. Middle: a camera fixed on the left arm's world target (red sphere). The number is the worse of the two arms' end-effector position error at that instant; the strip below is the same quantity over time. [MP4](media/hold_pose.mp4)*

With the arms locked, the end-effectors move with the mount: 31 mm RMS position error. The reactive controller removes 94% of that at 0.5 Hz and 63% at 3 Hz. Feeding forward the measured mount velocity cuts the remaining error by another 62% at 1.8 Hz.

## Setup

The mount is a mocap body standing in for the wearer's torso. Both arms are kinematic children of it, and each end-effector has a target pose fixed in the world frame. The mount follows a scripted six-axis motion, A sin(2πft) on each axis, with amplitudes taken from a treadmill-gait model at 1 m/s:

| axis | amplitude | frequency |
|---|---|---|
| x (forward) | 8 mm | f |
| y (left) | 22 mm | f/2 |
| z (up) | 20 mm | f |
| roll | 2° | f/2 |
| pitch | 1.3° | f |
| yaw | 3° | f/2 |

Only f changes between runs. The arms first settle on the static mount, then the disturbance starts and runs for 8 s at a 500 Hz control rate.

![The task](media/task_illustration.png)

*Four extreme mount poses of one cycle and (e) the average over the cycle: the mount and the proximal links blur, the end-effectors stay sharp. Motion enlarged 3x here so it shows in a still; all numbers use the unscaled motion.*

## Controller

![Control loop](media/control_loop.png)

Per arm, every 2 ms:

1. Compose the end-effector pose and Jacobian through the mount: world → mount → arm base → end-effector. The pose is never read from MuJoCo directly, so the same code can run on Vicon plus joint encoders.
2. PD on the world-frame pose and twist error gives a task twist.
3. Damped least squares maps it to joint velocities, with a null-space term that centres the joints.
4. A whole-arm safety filter finds the nearest joint velocity that keeps every link outside a cylinder around the wearer and respects joint speed and position limits ([docs/human-safety.md](docs/human-safety.md)).
5. Integrate to a joint position command and send it to the position servo.

Feedforward adds one term to step 2. The mount's contribution to the end-effector velocity is the measured total minus the arm's own J q̇; the controller commands the negative of it, so the arm moves before an error builds up instead of after.

## Results

Position and orientation error RMS over 8 s, worse of the two arms.

| f (Hz) | locked (mm) | reactive (mm) | removed | orientation (°) | peak joint speed (°/s) |
|---|---|---|---|---|---|
| 0.5 | 31.0 | 1.9 | 94% | 1.24 | 14 |
| 1.0 | 31.0 | 3.7 | 88% | 1.56 | 27 |
| 1.5 | 31.0 | 5.6 | 82% | 1.73 | 40 |
| 2.0 | 31.0 | 7.4 | 76% | 1.85 | 52 |
| 2.5 | 31.0 | 9.3 | 70% | 1.95 | 66 |
| 3.0 | 31.0 | 11.3 | 63% | 2.02 | 80 |

![Frequency sweep](media/freq_sweep.png)

*(a) Vertical mount motion in the slowest and fastest run. (b, c) End-effector error against disturbance frequency. Open marker: joint-speed limit reached (16% of samples at 3 Hz).*

The error grows roughly linearly with frequency. That is what a PD loop with a fixed bandwidth does: it only acts on error that already exists, so a faster disturbance leaves more of it behind. Orientation is harder than position; the controller removes 54% of the orientation error at 0.5 Hz and 25% at 3 Hz.

![Feedforward](media/feedforward.png)

*Right arm at f = 1.8 Hz. Position error RMS 6.7 → 2.6 mm (62% lower), orientation 1.79° → 1.25°.*

## Limitations

- The disturbance is a scripted sum of sinusoids, not recorded gait. It is periodic and smooth, which flatters any controller.
- In simulation the feedforward gets the exact mount velocity. On hardware it would come from Vicon, with noise and delay, and the gain would shrink.
- The arms are position-servoed and rigid: no link flexibility, backlash or servo dynamics beyond what MuJoCo models.
- The thesis direction was predictive control (MPC) against this baseline. That was not built; this repo is the reactive baseline and the feedforward extension.

## Code

| Path | |
|---|---|
| `controller/reactive_controller.py` | The whole control law, equations in order: errors, PD, feedforward, DLS, null space, safety projection |
| `controller/frames.py`, `pin_fk.py` | World-frame kinematics composed through the mount (Pinocchio FK) |
| `controller/human_safety.py`, `link_spheres.py` | 18 conservative spheres per arm, torso-frame keep-out constraints |
| `controller/runner.py`, `servo.py` | One control cycle in a fixed order, behind a takeover / exchange / release backend interface |
| `sim/` | MJCF scene, MuJoCo backend, scripted mount motion |
| `tools/disturbance_freq_sweep.py` | Frequency sweep (results table and figure) |
| `tools/walk_ff_compare.py` | Reactive vs feedforward comparison |
| `tools/make_readme_media.py` | The video above |
| `analysis/` | Validation scripts: FK against MuJoCo, end-effector velocity ([docs/velocity-validation.md](docs/velocity-validation.md)), gain sweeps, live dashboard |
| `planning/`, `examples/gpmp2_joint_space/` | Collision-aware Cartesian path planning and a GPMP2 joint-space demo |
| `cpp/` | C++20 port of the simulation and controller, matched to the Python golden trace at 1e-12 when ported ([cpp/README.md](cpp/README.md)) |

Two FK implementations exist on purpose: `pin_fk.py` (Pinocchio) is the control path, `kinematics.py` (analytical, from MuJoCo model constants) cross-checks it.

Frames are written `T_A_B`: pose of frame B in frame A (W world, T torso/mount, K arm base, E end-effector). Everything is SI internally; millimetres only in prints and plots.

## Running

Python 3.14 with `mujoco`, `pin`, `numpy`, `scipy`, `osqp`, `matplotlib` (`requirements.txt`; exact versions in `requirements-primary.txt`). Run from the repo root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mjpython main.py both                       # viewer (macOS needs mjpython)
python tools/disturbance_freq_sweep.py      # results table and sweep figure
python tools/walk_ff_compare.py right --speed=1   # feedforward at f = 1.8 Hz
python tools/make_readme_media.py           # README video
python -m unittest discover tests
```

Gains, limits, targets and the safety envelope live in `config/control.toml`.
