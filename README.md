# MSc Project — SRL MuJoCo Simulation

MuJoCo simulation of a torso-mounted dual Kinova Gen3 (supernumerary robotic
limbs). Current phase: a **reactive baseline controller** that holds a
world-frame end-effector pose while the torso (the human) moves —
"chicken-head" stabilization. Predictive control comes later, measured
against this baseline.

## Pipeline

Per control step, all SI (meters, radians; mm only at prints/plots):

```
target mocap pose (world) + joint angles qpos      sim/targets, MuJoCo
  -> FK EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)  controller/frames (Pinocchio FK)
  -> pose error  e = ref - actual (world)          controller/servo
  -> commanded twist v = Kp*e  [m/s; rad/s]        controller/servo (P law)
  -> joint rates qdot via damped least squares     controller/servo (DLS)
  -> integrate position-servo setpoints (rad)      controller/servo -> data.ctrl
  -> mj_step
```

The EE pose is never read from MuJoCo in the control path — it is composed
from the torso pose (future: Vicon) and arm FK (future: joint encoders),
mirroring what the hardware will provide.

## Setup

Requires the project virtual environment (Python 3.14 with `mujoco`,
`pinocchio`, `numpy`, `matplotlib`):

```bash
source .venv/bin/activate
```

If `.venv` is missing, recreate it from the Python.org framework install:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv --system-site-packages .venv
```

**macOS note:** anything that opens the MuJoCo viewer must be run with
`mjpython` (installed with the `mujoco` package), not plain `python` —
`launch_passive` needs the main thread on macOS. Run everything from the
repo root (model paths are CWD-relative).

## Running

Closed-loop simulation with viewer (controlled arms selectable):

```bash
mjpython main.py [right|left|both]   # default: both
```

Live position-error plot, one arm, headless (plain `python` is fine —
no MuJoCo viewer; close the plot window to stop and save the figure):

```bash
python -m plotting.right.position_error
python -m plotting.left.position_error
```

FK validation — MuJoCo's directly measured right EE position vs. the
independently composed FK (world → torso → Kinova base → EE):

```bash
python -m plotting.right.fk_validation
```

Tests:

```bash
python -m unittest discover tests
```

## Layout

```
main.py                    viewer loop: closed-loop world-frame pose hold
sim/
  scene.xml                MJCF scene: torso mocap body + dual Kinova Gen3 + targets
  world.py                 model/data singletons, checked id lookups
  targets.py               set/read EE target poses (mocap spheres, world frame)
  assets/kinova_gen3/      vendored Kinova Gen3 model
controller/
  transforms.py            pure SE(3)/rotation math (NumPy only)
  pin_fk.py                Pinocchio FK for one arm: T_K_E(q)   [control path]
  kinematics.py            analytical FK from MjModel constants [test reference only]
  frames.py                world-frame EE pose + Jacobian: T_W_T · T_T_K · T_K_E
  desired_pos.py           desired EE poses; resolved to world targets once
  servo.py                 the controller: P law + DLS (pure math) and the MuJoCo
                           plumbing (errors, setpoint integration, data.ctrl)
plotting/
  live_plot.py             generic live time-series plot (no MuJoCo)
  position_error.py        shared live position-error plot implementation
  right/, left/            per-arm entry points (position_error, fk_validation)
tests/                     unit + closed-loop tests (python -m unittest discover tests)
```

Two FK implementations exist deliberately: `pin_fk.py` (Pinocchio) is the
control path; `kinematics.py` (analytical, from MjModel constants only) is
an independent cross-check that catches regressions in the Pinocchio path.
Do not unify them.

## Conventions

- Frames: `T_A_B` denotes the pose of frame B expressed in frame A.
  W = world, T = torso, K = Kinova arm base, E = end-effector site.
- Poses are passed as `(pos (3,) meters, R 3x3)` pairs; quaternions are
  MuJoCo order `[w, x, y, z]`.
- All internal math is SI (meters, radians). Millimetres appear only at
  human-facing boundaries (prints, plots).
- End-effector references are world-frame: resolved against the torso once
  at startup, then held fixed in the world while the base moves.
