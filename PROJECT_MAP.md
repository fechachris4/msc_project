# Project Map

Audit date: 2026-07-07; architecture and evidence status refreshed
2026-07-23 after migration Steps 1–6. Current suite: 188 tests pass
(`python -m unittest discover tests`), plus the 500-cycle golden trace at
`rtol=1e-12`, `atol=1e-12`.

## 1. System Goal

Hold a **world-frame end-effector pose** with a torso-mounted dual Kinova Gen3
(supernumerary robotic limbs) while the torso (the human) moves underneath it —
"chicken-head" stabilization. Current phase: the strongest possible **reactive**
baseline in MuJoCo, with error plots (mean, RMSE, peak) good enough for the
thesis. Predictive/MPC work comes later and is explicitly out of scope now.

The sensing/estimation story: EE pose is computed as
`T_W_E = T_W_T (torso, from mocap) · T_T_K (fixed mount) · T_K_E(q) (arm FK)`,
never read directly from MuJoCo — mirroring what will be available on hardware
(Vicon torso pose + joint encoders).

## 2. Current Pipeline

```
sim/scene.xml + sim/assets/kinova_gen3/gen3.xml   (MJCF: torso mocap + 2 attached Gen3)
├── config/control.toml     (gains, limits, nominal dt, startup targets)
├── runtime_config.py       (strict immutable TOML loader)
├── controller/state.py       (fixed-shape SI records; Python/C++ boundary)
├── controller/backend.py     (takeover / exchange / release contract)
├── sim/world.py              (MujocoBackend: model, state, stepping, lifecycle)
├── controller/frames.py      (PlantState + framed targets → world controller state)
│   ├── controller/pin_fk.py       (Pinocchio T_K_E(q) — CONTROL PATH)
│   └── controller/transforms.py   (pure SE(3) math)
├── controller/reactive_pose.py        (pure PD + DLS + null-space policy)
├── controller/position_actuation.py   (limits + persistent position integration)
├── controller/servo.py                (explicit reactive-position composition)
├── controller/runner.py               (frame boundary, pipeline step, backend exchange)
├── controller/desired_pos.py          (configured framed targets + marker display)
├── main.py                             (viewer loop through Runner/backend)
├── plotting/                           (reusable instruments only)
└── analysis/                           (experiments and evidence)
    ├── live.py                         (Runner-based experiment + saved artifacts)
    ├── metrics.py                      (mean/RMSE/peak definitions)
    ├── reactive_baseline.py            (canonical baseline evidence)
    └── base_vs_error.py                (Runner-based live/headless trace)
```

## 3. Module Status Table

| Module/File | Purpose | Status | Depends On | Used By | Tests | Confidence | Touch? |
| ----------- | ------- | ------ | ---------- | ------- | ----- | ---------- | ------ |
| `sim/scene.xml` | Torso + dual Gen3 scene, targets, contact excludes | Validated | `assets/kinova_gen3/gen3.xml` | `world.py` | Loads in all tests | High | Touch only if needed |
| `config/control.toml` | Shared startup gains, limits, nominal dt, and targets | Implemented | — | `runtime_config.py` | `test_runtime_config.py` + golden trace | High | Single tuning surface |
| `runtime_config.py` | Strict immutable TOML loader and effective-config stamp | Implemented | control.toml | controller, sim, analysis | `test_runtime_config.py` | High | Keep Python/C++ schema aligned |
| `sim/assets/kinova_gen3/` | Vendored Gen3 MJCF (position-servo actuators) | Validated (kinematics + closed loop) | — | scene, `pin_fk.py` | Indirect via FK tests | High | Touch only if needed |
| `sim/world.py` | `MujocoBackend`: model/data ownership, checked ids, state read, command apply, stepping, lifecycle | Implemented | scene.xml, state records | Runner, viewers, ground-truth tests | `test_runner_backend.py` + integration suite | High | Backend-specific code only |
| `sim/targets.py` | Display target mocap poses | Implemented | `world.py` | desired_pos | `test_reference.py` | Medium-High | Never a controller input |
| `sim/motion.py` | Matched scripted torso pose/twist functions | Validated | transforms | MuJoCo backend drivers, experiments | `test_motion`: pose/twist oracle, zero-guard, closed-loop rejection | High | `main.py` uses module defaults; experiments own `ExperimentConfig` |
| `controller/transforms.py` | Pure SE(3)/rotation math | Validated | numpy | kinematics, frames, desired_pos | `test_kinematics.RotationHelpersTest`, `test_transforms` (vs Pinocchio) | High | Do not touch |
| `controller/kinematics.py` | Analytical FK from MjModel constants | Validated, **demoted to test reference** | transforms | `test_pin_fk.py` only | `AnalyticalFKTest` vs MuJoCo, 50 cfg/arm, 1e-9 | High | Do not touch |
| `controller/pin_fk.py` | Pinocchio-backed `T_K_E(q)` — control path FK | Validated | pinocchio, `gen3.xml` | `frames.py` | `test_pin_fk` vs analytical FK, 1e-9 | High | Do not touch |
| `controller/state.py` | Fixed-shape, read-only SI plant/target/command records | Implemented | numpy | backend, frames, Runner, pipelines | `test_state_frames.py` | High | Keep one-to-one with future C++ structs |
| `controller/frames.py` | Pure boundary from plant/framed target to world controller quantities | Validated (incl. moved torso) | state, pin_fk, transforms | Runner, diagnostics | `WorldFrameEETest`, `JacobianWorldTest`, `test_state_frames.py` | High | No backend imports or controller flags |
| `controller/desired_pos.py` | Configured framed targets and MuJoCo marker display | Implemented | state, transforms, targets | `main.py`, analysis | `test_reference.py` | High | No control math |
| `controller/reactive_pose.py` | Pure world-frame PD, DLS IK, and null-space policy | Validated | state, runtime config | servo pipeline | golden trace + `test_reactive_pipeline.py` | High | Sole numerical implementation |
| `controller/position_actuation.py` | Joint-velocity clipping and persistent position integration | Validated | state | servo pipeline | golden trace + `test_reactive_pipeline.py` | High | Reconstruct to reset |
| `controller/servo.py` | Explicit reactive-pose-to-position pipeline composition | Validated | reactive_pose, position_actuation, state | Runner, tests | golden trace + pipeline/closed-loop tests | High | No frames, backend, or MuJoCo |
| `controller/backend.py` | Minimal plant protocol: takeover, exchange, release | Implemented | state | Runner, backends | `test_runner_backend.py` | High | Do not add backend-specific operations |
| `controller/runner.py` | Current cycle ordering and reconstruction ownership | Validated | backend, frames, servo | main, live, base_vs_error | `test_runner_backend.py` + closed-loop tests | High | Explicit pipeline, no mode flags |
| `main.py` | Viewer loop through `ReactivePositionRunner` | Implemented | desired_pos, runner, world backend | user | entry-point + Runner tests | High | No direct control math |
| `plotting/live_plot.py` | Generic live plot, expand-only y-autoscale | Implemented | matplotlib | base_vs_error, fk_validation | None (visually exercised) | Medium | Do not touch |
| `plotting/style.py` | Shared Okabe-Ito colors + side→color/linestyle maps | Implemented | — | dashboard, base_vs_error, diagnose, validate_velocity, bandwidth_sweep | None (constants only) | High | Touch only if needed |
| `analysis/metrics.py` | One metric definition: `stats`/`print_stats` (full-run) + `windowed_stats` | Implemented | numpy, world | base_vs_error, dashboard | Hand-verified: windowed_stats matches stats() on identical data | High | Touch only if needed |
| `analysis/dashboard.py` | Live 7-panel control-loop dashboard (per-axis error, e_rot, headroom, σ_min, margins) | Implemented | frames, servo, metrics | user | `test_dashboard.py`: pure per-tick helpers + immutable-config routing | Medium-High | Touch only if needed |
| `analysis/base_vs_error.py` | Thesis success-criterion live/headless trace through Runner/backend | Implemented | runner, live_plot, metrics | user | Closed-loop tests + Step 6 run | High | Monitoring path |
| `analysis/reactive_baseline.py` | Provenance-stamped mean/RMSE/peak baseline artifact | Implemented | live, metrics, runtime config | thesis evidence | `test_reactive_baseline.py` + Step 6 canonical run at `e0fbb74` | High | Primary baseline evidence |
| `analysis/fk_validation.py` | Live direct-vs-FK comparison | Implemented | frames, world, live_plot | user | Visual only | Medium | Touch only if needed |
| `README.md` | Setup + pipeline + layout docs | Current (refreshed 2026-07-07) | — | — | — | High | Keep in sync |
| `tasks/todo.md` | Historical task log | Stale (history, not current state) | — | — | — | — | Do not touch (append-only) |
| `docs/superpowers/` | FK design spec + plan (2026-07-06) | Done, matches code | — | — | — | High | Do not touch |
| `requirements.txt` | mujoco, numpy, pin | Incomplete (**matplotlib missing**) | — | — | — | Medium | Needs work (one line) |

## 4. Validated Components

1. **Rotation/transform math (`controller/transforms.py`)** —
   `test_kinematics.RotationHelpersTest` (1e-12) plus `test_transforms`
   cross-validation against Pinocchio. Frozen.

2. **Analytical FK (`controller/kinematics.py`)** — independent of MuJoCo pose
   outputs; 50 random configs/arm vs `mj_kinematics` at 1e-9. Frozen; exists to
   cross-check other FK implementations. Do not "improve" it.

3. **Pinocchio FK (`controller/pin_fk.py`)** — `pin_T_K_E(q)` vs analytical FK
   at 1e-9; base_link-at-identity assertion guards frame validity. Frozen.

4. **World-frame EE pose (`controller/frames.py`)** —
   `T_W_T · T_T_K · T_K_E(q)` matches MuJoCo's world site pose at 1e-9, both at
   the nominal torso pose and across 50 random torso translations+rotations
   (`WorldFrameEETest`, incl. moved-torso cases). The static-torso-only caveat
   from the original audit is resolved.

5. **World-aligned Jacobian (`controller/frames.py`)** — 6x7, rows
   [v (m/s); ω (rad/s)], validated against `mj_jacSite` at 1e-9 across random
   configurations and random torso poses (`JacobianWorldTest`).

6. **Reactive controller pipeline** —
   `reactive_pose.py` owns the pure PD/DLS/null-space law;
   `position_actuation.py` owns clipping and persistent position integration;
   `servo.py` composes only those two stages. The frozen 500-cycle trace and
   closed-loop tests validate the split without a numerical change.

7. **Reference resolution (`controller/frames.py`)** —
   world/base/torso `FramedTarget` records are resolved into `WorldTarget`
   records at the Runner boundary on every cycle. The controller receives only
   world-frame pose/twist quantities and contains no frame conditionals.

8. **Scene mounting (`sim/scene.xml`)** — arm mounts at torso-relative
   `(-0.16, ±0.10, 0.14)` with ±0.845708 rad roll; contact excludes for the
   mocap-welded base. Frozen unless the physical backpack geometry changes.

## 5. Incomplete Components

1. **Default-scenario baseline quality** — logging and mean/RMSE/peak artifacts
   now exist, but the current default sway exposes large left-arm error and
   torso contact. Treat this as a controller/task-geometry result to diagnose,
   not as a reason to widen metrics or silently retune.

2. **Known residual at the left task point** — the left `desired_pos` pose
   sits at joint_6's ctrl limit; the clip leaves a ~6 mm steady-state residual
   there by design, not a controller bug (see `ClosedLoopConvergenceTest`
   docstring). Revisit the task point or mount if it matters for the thesis
   scenario.

3. **Small hygiene** — `requirements.txt` lacks `matplotlib`.

## 6. Dependency Graph

```
config/control.toml -> runtime_config.py
sim/scene.xml -> sim/world.MujocoBackend -> PlantState / JointPositionCommand
                                  ^                   |
                                  | exchange          v
source FramedTargets -> frames -> ReactivePositionRunner
                                  |
                                  v
                  ReactivePositionPipeline
                  ├── ReactivePoseController
                  └── PositionIntegrator

main.py / analysis.live / analysis.base_vs_error use the same Runner path.
MuJoCo model/data remain available separately for viewers, telemetry, and
ground-truth tests; controller code never reads them.
```

Note the deliberate redundancy: `kinematics.py` (analytical) and `pin_fk.py`
(Pinocchio) both compute `T_K_E`; the analytical one exists to catch regressions
in the Pinocchio path and must stay independent of it.

## 7. Research Readiness

| Subsystem | Implemented | Validated | Paper-ready |
| --- | --- | --- | --- |
| Scene / robot model | ✅ | ✅ | ✅ (actuation: position servos, recorded) |
| Transform math | ✅ | ✅ | ✅ (nothing to publish, just trustworthy) |
| Arm FK (both impls) | ✅ | ✅ (1e-9, dual-implementation cross-check) | ✅ as methodology |
| World-frame EE pose + Jacobian | ✅ | ✅ (incl. moved torso, vs mj_jacSite) | ✅ as methodology |
| Reference/targets | ✅ | ✅ | ✅ |
| Reactive controller | ✅ | ✅ (golden trace + closed-loop + Runner/backend) | ⚠️ default-scenario performance needs improvement |
| Base motion scripting | ✅ | ✅ (`test_motion` oracle + rejection) | ✅ |
| Error metrics + plots | ✅ | ✅ (mean/RMSE/peak persistence + artifact test) | ✅ as baseline evidence |

Blunt read: estimation/kinematics, the explicit controller, backend exchange,
scripted base motion, and evidence pipeline are validated. The remaining
reactive-phase work is improving or honestly bounding performance under the
default experiment scenario, especially the left-arm contact/error, before it
is used as the comparison baseline.

## 8. Next Work Queue

1. **Diagnose default-scenario left-arm performance** — diff against the last
   working commit and check task geometry, frames, joint limits, and contact
   before gain changes.
2. **Parameter sweep hooks** — run the existing `ExperimentConfig` scenario
   parameters through the same Runner and metrics pipeline over at least two
   disturbance conditions.

Actuation decision (recorded): the reactive baseline commands **position-servo
setpoints via resolved-rate differential IK**. `PositionIntegrator` advances
the command from its measured-state seed; `MujocoBackend` alone writes
`data.ctrl`. Torque control would require a separate explicit pipeline and
backend command contract.

## 9. Do-Not-Redo List

- `controller/transforms.py` — pure math, tested to 1e-12. Done.
- `controller/kinematics.py` — analytical FK, deliberately demoted to test
  reference. Do not delete, do not port to Pinocchio, do not "unify" the two
  FK implementations — the redundancy is the verification strategy.
- `controller/pin_fk.py` — done and frozen.
- `controller/frames.py` composition (`T_W_T · T_T_K · T_K_E`) and Jacobian —
  validated including moved torso; matches the hardware sensing story.
- `controller/reactive_pose.py` is the sole PD/DLS/null-space numerical law.
  Do not duplicate it in a backend, diagnostic, or future controller wrapper.
- The circular-FK investigation — found, fixed, documented (2026-07-06 spec).
- `sim/scene.xml` mount geometry and contact excludes — working.
- `plotting/live_plot.py` — good enough for live monitoring; thesis figures are
  a separate offline path, not an upgrade of this.

## 10. Risks and Unknowns

1. **Mocap base motion is kinematic.** MuJoCo does not derive torso velocity
   from mocap writes. The simulation scenario therefore supplies matched
   analytic torso pose and twist to `MujocoBackend`; a future Vicon boundary
   must instead supply synchronized, timestamped measurements.
2. **Actuator model mismatch.** Simulated Gen3 uses MuJoCo position servos with
   vendored gains; the real Gen3 exposes different low-level interfaces. Gains
   will not transfer — only the structure will.
3. **Torso is kinematic (mocap): no reaction forces.** The human doesn't feel
   the arms and arm inertia can't destabilize the base. Intended phase-1
   abstraction; on hardware the backpack dynamics are two-way. Don't over-tune
   to this idealization.
4. **Import-time default backend.** `sim.world` constructs the default MuJoCo
   backend at import for the current scripts and tests. Asset paths are
   module-relative, so startup is CWD-independent; the later C++ port should
   keep construction explicit.
5. **Two FK configs must stay in lockstep.** `pin_fk` loads `gen3.xml` directly;
   the scene attaches the same file. The analytical-FK cross-test is the
   tripwire — run tests after any `gen3.xml` edit.
6. **Target mocap bodies are display-only.** `desired_pos.apply()` returns
   retained framed targets and initializes their markers; the Runner resolves
   the retained records, never reads marker state back as control input.

Resolved since the original audit: moved-torso coverage of `T_W_T` (now
tested), missing Jacobian (added + validated), empty controller (implemented +
validated), `torso_mocap_id` naming hazard (renamed `torso_body_id`), stale
README (refreshed), and implicit target-marker state (replaced by retained
framed targets resolved explicitly at the Runner boundary each cycle).

## 11. Extending the Sim

Four concrete change rules for the current simulation scope.

1. **New trajectory shape** (circle, figure-eight, square wave): implement
   it as an `offset(t)`/`rate(t)` pair in `controller/transforms.py`,
   analytic derivatives of each other (like `sine_offset`/`sine_rate`), then
   swap the pair into `sim/target_motion.py`'s marked EXTENSION POINT —
   `target_pose_at` and `target_twist_at` together. An offset/rate pair that
   isn't an exact derivative breaks velocity validation silently; no test
   fails at the pose level.

2. **New disturbance** (chirp, step, recorded gait playback): provide matched
   `pose_at(t)` and `twist_at(t)` functions and install both on
   `MujocoBackend.configure_torso_driver`. They must describe the same
   trajectory at the same simulation time.

3. **New controller**: add its mathematical policy as a pure module, then add
   one explicit pipeline/Runner composition matching its real data flow.
   Reuse `PlantState`, world-frame resolution, backend exchange, configuration,
   and telemetry. Do not add a controller-mode flag or force a controller
   through IK or position integration if it bypasses that stage.

4. **Disabling features**: base motion off = install the matched motion
   functions with zero amplitudes (or no torso driver); target motion off =
   retain the configured static targets; one arm only = the `main.py` CLI arg
   (`right`/`left`/`both`). The fixed two-arm records are deliberate scope,
   not a general limb registry.
