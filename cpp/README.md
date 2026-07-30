# C++ port of the MSc SRL MuJoCo simulation

A modern C++20 port of the Python simulation in the parent directory: a
torso-mounted dual Kinova Gen3 (supernumerary robotic limbs) holding a
world-frame end-effector pose while the torso moves — "chicken-head"
stabilisation.

The Python implementation is **unmodified**. This directory is additive.

Documentation, in reading order:

| Document | Contents |
|---|---|
| [`docs/01-architecture.md`](docs/01-architecture.md) | How the Python simulation works: entry point, startup order, cycle order, mathematics, outputs |
| [`docs/02-behaviour-contract.md`](docs/02-behaviour-contract.md) | The clause-by-clause contract the port must satisfy (A1–M4) |
| [`docs/03-cpp-design.md`](docs/03-cpp-design.md) | C++ architecture, dependencies, numerics strategy, staged migration plan |
| [`docs/04-parity-report.md`](docs/04-parity-report.md) | Measured parity results, where residuals come from, what is still unvalidated |

## What was ported

The **simulation**: entry point, configuration, model loading, the control
loop, trajectory generation, the controller, state updates, rendering and
telemetry.

| Python | C++ | Notes |
|---|---|---|
| `runtime_config.py` | `src/config/RuntimeConfig.*` | Strict TOML loader; byte-identical effective-config output |
| `controller/state.py` | `src/core/Types.h` | Plain aggregates, one per Python record |
| `controller/transforms.py` | `src/math/Transforms.*` | Pure SE(3)/SO(3) |
| (NumPy `linalg`) | `src/math/LinAlg.*` | Accelerate `dgesv`/`dgesdd` — the same LAPACK NumPy uses |
| `controller/pin_fk.py` | `src/kinematics/PinModel.*` | Same Pinocchio 4.0.0 binary |
| `controller/frames.py` | `src/kinematics/Frames.*` | World-frame kinematics + target resolution |
| `controller/link_spheres.py` | `src/kinematics/LinkSpheres.h` | **Generated** by `tools/generate_link_spheres.py` |
| `controller/human_safety.py` | `src/control/HumanSafety.*` | Torso-frame envelope geometry |
| `controller/reactive_controller.py` | `src/control/ReactiveController.*` + `SafetyFilter.*` | Equations 1–6 and equation 7 respectively |
| `controller/position_actuation.py` | `src/control/PositionActuation.*` | Persistent position integration |
| `controller/cylinder_router.py` | `src/control/CylinderRouter.*` | Keep-out routing |
| `controller/servo.py` | `src/control/Servo.*` | Pipeline composition |
| `controller/runner.py` | `src/control/Runner.*` | Cycle ordering |
| (new optimized-plan boundary) | `src/planning/JointTrajectory.*`, `PlanValidation.*` | Time-stamped joint states, Hermite sampling, exact post-validation |
| HumanSL GPMP2 prototype | `src/planning/Gpmp2Planner.*` | Optional joint-space factor graph adapter |
| (new optimized-plan executor) | `src/control/JointTrajectoryController.*`, `PlannedJointRunner.*` | Joint feedback, existing safety projection and position integration |
| `controller/trajectory.py` | `src/trajectory/Trajectory.*` | Hold / C² waypoint spline / circle / program / periodic |
| `controller/trajectory_config.py` | `src/trajectory/TrajectoryConfig.*` | Structured-intent compiler |
| `controller/backend.py` | `src/sim/Backend.h` | Takeover / exchange / release |
| `sim/world.py` | `src/sim/MujocoBackend.*` | Model/data ownership, stepping, lifecycle |
| `sim/motion.py`, `sim/target_motion.py` | `src/sim/Motion.*`, `TargetMotion.*` | Scripted disturbances |
| `sim/targets.py`, `controller/desired_pos.py` | `src/sim/Targets.*`, `DesiredPos.*` | Marker I/O and configured targets |
| `sim/target_trajectory.py` | `src/sim/TargetTrajectory.*` | Trajectory initialisation |
| `sim/cylinder_view.py`, `sim/human_safety_view.py` | `src/render/Overlays.*` | Viewer geometry |
| `mujoco.viewer.launch_passive` | `src/render/Viewer.*` | GLFW + `mjr` |
| `main.py` | `src/app/main.cpp` | Viewer entry point |
| `tests/golden_trace.py` | `src/app/golden_trace.cpp` + `src/telemetry/TraceWriter.*` | 210-column trace, identical schema |

## Intentional deviations

Each is a deliberate choice, not an oversight.

1. **No import-time global backend.** `sim/world.py` constructs a
   `MujocoBackend` at module import; the C++ constructs everything explicitly
   in `main`. Same behaviour — and what the Python project's own risk list
   (`PROJECT_MAP.md` §10.4) asks a C++ port to do.
2. **Validation at boundaries, not on every temporary.** The Python records
   re-validate shapes and finiteness on every construction. The port
   validates at the boundaries that can actually produce bad data (config
   load, backend read, target sampling) rather than 500 times a second on the
   control path.
3. **The `analysis/` and `plotting/` suite is not ported.** Those are
   matplotlib experiment tooling, not the simulation. The port emits the same
   underlying numbers as CSV (`srl_headless_trace`, `srl_golden_trace`) so the
   existing Python figures can be produced from C++ output.
4. **The viewer is GLFW + `mjr`**, not `mujoco.viewer.launch_passive`, which
   is Python-only. Scene, camera defaults and overlay geometry match;
   interaction affordances differ (left-drag rotate, right-drag pan, scroll
   zoom, Esc to quit). `--trajectory-plot` is accepted and reported as
   unsupported rather than silently ignored.
5. **OSQP is a build option** (`-DSRL_WITH_OSQP=ON` by default). The harness
   reports `qp_fallback_entries` so the effect of omitting it is measured, not
   assumed — it is currently 0 in every scenario tested.
6. **Legacy `shape = "measured_start_displacement"` trajectory tables are
   rejected** with an explicit error instead of being translated. The shipped
   config uses the segment form; the legacy shim exists only for backwards
   compatibility in the Python.
7. **`LinkSpheres.h` is generated, not transcribed** — 18 spheres of
   17-significant-digit constants are too easy to corrupt by hand.

## Build

Requirements: CMake ≥ 3.24, a C++20 compiler, LAPACK (Apple Accelerate on
macOS or a system LAPACK on Linux), and the project's Python venv present at
`../.venv` — the build links the native MuJoCo and Pinocchio libraries that
venv already ships.

Eigen, toml++, GLFW, urdfdom_headers and OSQP are fetched at configure time
from pinned versions into the build tree. Nothing is installed system-wide.

```bash
cd ..                                  # the Python project root
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build -j
```

Options: `-DSRL_WITH_VIEWER=OFF`, `-DSRL_WITH_OSQP=OFF`,
`-DSRL_WITH_GPMP2=OFF`,
`-DSRL_BUILD_TESTS=OFF`, `-DSRL_VENV=/path/to/.venv`.

## Optional GPMP2 joint-space planning

The planner is an optional layer and is off by default. Its execution flow is:

```text
measured q, qdot + joint goal + torso-frame human SDF
    -> HumanSL straight-line TrajectoryInitiation equivalent
    -> HumanSL GPMP2 factor graph (GP, limits, SDF, self-collision)
    -> control-rate interpolateArmTraj densification
    -> time-stamped q, qdot JointTrajectory conversion
    -> exact Pinocchio/joint-limit post-validation
    -> qdot_ref + Kp(q_ref - q_measured)
    -> existing whole-arm safety projection
    -> existing persistent position integration
    -> MuJoCo backend
```

The GPMP2 bundle currently available in `HumanSL_MAIN/third_party` contains
Linux x86-64 `.so` files. Build and run this option on the Ubuntu workspace,
not on macOS:

```bash
cmake -S cpp -B cpp/build \
  -DSRL_WITH_GPMP2=ON \
  -DSRL_GPMP2_ROOT=/path/to/HumanSL_MAIN/third_party \
  -DSRL_WITH_VIEWER=ON
cmake --build cpp/build -j

./cpp/build/srl_gpmp2_plan left planned.csv \
  -0.70 0.45 -0.20 1.20 -0.20 0.75 0.30

./cpp/build/srl_gpmp2_sim left \
  -0.70 0.45 -0.20 1.20 -0.20 0.75 0.30
```

This is a selective port of HumanSL_MAIN's live source chain:
`Gen3Arm::plan_joint` -> `TrajectoryInitiation` -> `OptimizeTrajectory` ->
`densifyTrajectory`/`interpolateArmTraj` -> `convertTrajectory`. It reuses
HumanSL's Kinova DH model, 34-sphere `GenerateArmModel` geometry,
self-collision pairs, GP priors, joint/velocity limit costs, support-point and
interpolated SDF factors, Levenberg-Marquardt structure, and separate
control-rate densification. HumanSL's Vicon/C3D input and hardware execution
are deliberately replaced at their boundaries by the simulator's measured
state, torso-cylinder SDF, exact validation, and existing MuJoCo joint runner.

Both applications take seven goal joint angles in radians. Planning happens
on a background thread while the visible MuJoCo simulation runs at its 2 ms
control period and holds the measured joints. `srl_gpmp2_plan` remains a
CSV-only diagnostic. `srl_gpmp2_sim` starts the viewer immediately, refuses
to execute unless every interpolated 2 ms sample passes the simulator's real
joint-limit and 18-sphere human-clearance check, and then hands the accepted
plan to the live joint controller. The controller is paced toward 500 Hz
while rendering is limited to roughly 60 Hz.

This is one in-simulation planning request from the command-line goal, not
continuous replanning. The loop targets the configured 2 ms wall-clock period
but is not a hard real-time scheduler; an over-budget cycle runs late rather
than skipping physics.

The GPMP2 cost uses HumanSL's 34-sphere DH model, but that historical model is
still not accepted as collision proof for this simulator. `PlanValidation`
independently checks the current Pinocchio 18-sphere model at every 2 ms
sample, and `PlannedJointRunner` applies the existing safety filter again
online. Any disagreement therefore rejects or stops the plan safely.

## Run

```bash
./cpp/build/srl_sim                 # viewer; arm selection from control.toml
./cpp/build/srl_sim right           # or left / both
./cpp/build/srl_sim --planning-smoke # plan + 50 closed-loop cycles, no viewer
./cpp/build/srl_print_config        # effective configuration + sha256
./cpp/build/srl_smoke               # dependency/linkage check

./cpp/build/srl_golden_trace   --out trace.csv
./cpp/build/srl_headless_trace --out headless.csv --steps 2000
./cpp/build/srl_trajectory_dump cpp/tests/fixtures/trajectory_control.toml 400
```

Tests:

```bash
ctest --test-dir cpp/build --output-on-failure
```

When `[planning].enabled = true`, `srl_sim` runs the native C++ equivalent of
the Python Cartesian planner at startup. It captures the current torso and
end-effector poses, optimises dense samples of the shared minimum-jerk spline
against the torso-frame human/box and world-floor distances, retimes the
result through the configured Cartesian limits, applies the optional lead
prefilter, and gives that WORLD-frame per-arm source to the unchanged reactive
runner. The planned spline and knots are drawn in the viewer. This path is
separate from the optional GPMP2 joint-space applications above.

## How behavioural parity was verified

One command runs the whole gate:

```bash
bash cpp/tools/verify_parity.sh
```

It builds, runs the 336 unit assertions, and then performs four
Python-vs-C++ comparisons at `rtol = atol = 1e-12` — the same tolerance the
Python golden trace holds itself to:

1. **Effective configuration** — byte-identical, including CPython float
   `repr()` and the config SHA-256.
2. **Golden trace** — 250 cycles × 2 arms against the *committed*
   `tests/golden/reactive_current.csv`: 105,000 fields, worst difference
   2.167e-13.
3. **Headless default-config trace** — 2,000 closed-loop cycles with human
   safety, cylinder routing and the configured trajectory all enabled:
   236,000 fields, worst difference 4.441e-13, and **every discrete column
   identical** (safety-ladder branch, stop flags, active constraint counts,
   route kinds).
4. **Trajectory generation** — hold + line + 4-point C² spline + 2-revolution
   circle: durations, boundaries, rate bounds and all rotational quantities
   bit-identical; translation residual ≤ 1.13e-14.

The residual traces to one-ULP differences in dense matrix products,
amplified by the conditioning of the damped-least-squares solve. Full
analysis and the list of what remains unvalidated are in
[`docs/04-parity-report.md`](docs/04-parity-report.md).
