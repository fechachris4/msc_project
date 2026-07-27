# MSc Project — SRL MuJoCo Simulation

MuJoCo simulation of a torso-mounted dual Kinova Gen3 (supernumerary robotic
limbs). Current phase: a **reactive baseline controller** that holds a
world-frame end-effector pose while the torso (the human) moves —
"chicken-head" stabilization. Predictive control comes later, measured
against this baseline.

## Pipeline

Per control step, all SI (meters, radians; mm only at prints/plots):

```
sample target source once + backend PlantState
  -> resolve target/state into world frame          controller/frames
  -> pose/twist error, PD, DLS, null-space qdot     controller/reactive_controller
  -> whole-arm human-distance safety projection     controller/reactive_controller
  -> clip and integrate joint-position command      controller/position_actuation
  -> apply command, mj_step, read next PlantState   sim/world.MujocoBackend
```

The EE pose is never read from MuJoCo in the control path — it is composed
from the torso pose (future: Vicon) and arm FK (future: joint encoders),
mirroring what the hardware will provide.
`ReactivePositionRunner` owns that ordering. Target mocap bodies are display
markers, not controller inputs.

Whole-arm human protection is enabled under `[human_safety]` in
`config/control.toml`. Conservative sphere chains cover the Kinova collision
meshes; their world positions and joint Jacobians are recomputed every cycle.
A torso-frame human envelope then constrains the requested joint velocity
before position integration. The older `[cylinder_keepout]` waypoint router
remains optional path shaping and is not the safety mechanism. The equation,
geometry derivation, stop behavior, and current evidence are documented in
[`docs/human-safety.md`](docs/human-safety.md).

## Setup

Requires the project virtual environment (Python 3.14 with `mujoco`,
`pinocchio`, `numpy`, `scipy`, `osqp`, `matplotlib`):

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
mjpython main.py                          # targets from config/control.toml
mjpython main.py [right|left|both]        # optional command-line override
mjpython main.py both --trajectory-plot   # optional nonblocking live EE paths
```

Each `[targets.right]` / `[targets.left]` table is always a target source.
Without a nested `.trajectory` table, its existing pose is held statically.
With timed trajectory data, `main.py` initializes the configured simulation
posture once, then samples the path through the real-time Runner boundary.
There is no hidden replay and no separate trajectory mode.

Base motion is configured by the levers at the top of `sim/motion.py`
(amplitude/frequency); zero amplitude = static base, and the torso then
stays hand-draggable in the viewer as an improvised perturbation.

The intended trajectory interface is conversational: describe the motion to
Codex, let it resolve the arm/frame/geometry/orientation and only genuine
ambiguities, then have it materialize structured data, report the resolved
parameters, and launch the viewer/live path plot.
For example:

- “Make the left hand trace a 10 cm diameter circle in the torso YZ plane,
  keep its orientation fixed, and repeat slowly.”
- “Move the left hand smoothly through offsets `[4,3,2]`, `[8,1,5]`, and
  `[12,0,2]` cm in world coordinates, respecting 0.1 m/s.”

There is no natural-language parser in the robotics runtime. The rigorous
internal vocabulary is hold, line, C2 waypoint spline, circle, and sequence.
The current user-configured displacement form remains compatible and is
translated into line segments internally. Remove the nested trajectory table
to return that arm to its existing fixed `position_m` / `rpy_rad`.
Simulation-only starting joints remain separate under
`[simulation.initial_joint_position_rad]`. The structured Codex/programmatic
seam and reproducibility examples are in `docs/cartesian-trajectories.md`.

Base motion vs. EE error, both arms, headless (plain `python` is fine —
no MuJoCo viewer; live mode opens a rolling plot and runs until the
window is closed; `--save T` runs headlessly for `T`
sim-seconds instead):

```bash
python -m analysis.base_vs_error
python -m analysis.base_vs_error --save 30
```

Provenance-stamped reactive-baseline validation with thesis-ready
mean/RMSE/peak and desired-vs-measured world-path figures:

```bash
python -m analysis.reactive_baseline
# A canonical evidence run requires a clean worktree and pinned environment:
python -m analysis.reactive_baseline --canonical --output /tmp/reactive-baseline
```

Live 7-panel control-loop dashboard, one or both arms (same `--save T`
convention):

```bash
python -m analysis.dashboard right
python -m analysis.dashboard both --save 10
```

FK validation — MuJoCo's directly measured right EE position vs. the
independently composed FK (world → torso → Kinova base → EE):

```bash
python -m analysis.fk_validation
```

Tests:

```bash
python -m unittest discover tests
```

## Layout

```
main.py                    viewer loop: closed-loop world-frame pose hold
runtime_config.py          strict immutable loader for shared control TOML
config/
  control.toml             arm, gains, limits, nominal dt, and startup targets
sim/
  scene.xml                MJCF scene: torso mocap body + dual Kinova Gen3 + targets
  world.py                 MuJoCo backend: model/data, exchange, lifecycle
  human_safety_view.py     viewer-only safety envelope and link-sphere display
  target_trajectory.py     trajectory initialization for the real-time loop
  targets.py               set/read EE target poses (mocap spheres, world frame)
  motion.py                scripted base motion: sinusoidal torso disturbance
  assets/kinova_gen3/      vendored Kinova Gen3 model
controller/
  backend.py               minimal takeover/exchange/release plant contract
  runner.py                explicit reactive pose-to-position cycle ordering
  trajectory.py            pure static/waypoint/program target sources
  trajectory_config.py     validated structured-intent compiler
  transforms.py            pure SE(3)/rotation math (NumPy only)
  pin_fk.py                Pinocchio FK for one arm: T_K_E(q)   [control path]
  kinematics.py            analytical FK from MjModel constants [test reference only]
  frames.py                target/state boundary + world-frame EE kinematics
  desired_pos.py           configured framed targets and MuJoCo marker display
  link_spheres.py          mesh-derived conservative whole-arm geometry
  human_safety.py          torso-frame distances and link-point constraints
  reactive_controller.py   complete control + safety equations, top-to-bottom
  position_actuation.py    velocity limits + persistent position integration
  servo.py                 explicit reactive-pose-to-position composition
plotting/                 reusable instruments only (no MuJoCo except via callers)
  live_plot.py             generic live time-series plot, expand-only autoscale
  live_cartesian_path.py   bounded nonblocking live EE-path publisher
  style.py                 shared Okabe-Ito colors + side conventions
analysis/                  every experiment script; figures -> analysis/output/
  metrics.py               one metric definition: stats/print_stats/windowed_stats
  reactive_baseline.py     stamped validation run + mean/RMSE/peak figure
  cartesian_path.py        aligned desired-vs-measured world-path figure
  dashboard.py             live 7-panel control-loop dashboard, one or both arms
  base_vs_error.py         thesis success-criterion figure: base disp vs EE error
  diagnose.py              pinned-scenario failure diagnosis
  bandwidth_sweep.py       reactive-loop tracking bandwidth vs frequency
  validate_velocity.py     ee_velocity vs finite-difference ground truth
  fk_validation.py         live direct-vs-FK comparison (right arm)
tests/                     unit + closed-loop tests (python -m unittest discover tests)
tools/
  derive_link_spheres.py   regenerate conservative geometry from collision meshes
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
- Controller math is world-frame. World/base/torso target selection is resolved
  at the Runner boundary every cycle; the configured baseline references are
  world-frame and therefore remain fixed while the base moves.
- Target-source time starts at backend takeover. Waypoint conventions, program
  frame rules, and the C++ replay fixture are documented in
  `docs/cartesian-trajectories.md`.
- Runtime control values come from `config/control.toml`. Edit that file and
  restart; the effective configuration and source hash are printed at startup
  and stamped into saved experiment metadata.
