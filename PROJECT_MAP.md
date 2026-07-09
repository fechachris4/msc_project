# Project Map

Audit date: 2026-07-07, refreshed later the same day after the reactive
controller landed (commits `6bfd9af`…`f29df5c`). Test suite at refresh time:
25/25 pass (`python -m unittest discover tests`). Working tree: clean.

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
└── sim/world.py            (MjModel/MjData singletons, checked id lookups)
    ├── sim/targets.py      (write/read target mocap poses, world frame)
    ├── sim/motion.py       (scripted base motion: sinusoidal torso disturbance)
    ├── controller/frames.py
    │   ├── controller/pin_fk.py       (Pinocchio T_K_E(q) — CONTROL PATH)
    │   ├── controller/transforms.py   (pure SE(3) math)
    │   ├── → world-frame EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)
    │   └── → world-aligned 6x7 Jacobian (validated vs mj_jacSite)
    ├── controller/kinematics.py       (analytical FK — TEST REFERENCE only)
    ├── controller/desired_pos.py      (desired EE poses → world targets, once)
    ├── controller/servo.py            (controller: e → v = Kp*e → qdot via DLS → apply_ctrl)
    ├── main.py                        (viewer loop: closed loop via servo.apply_ctrl)
    ├── plotting/                      (reusable instruments only)
    │   ├── live_plot.py               (generic live time-series plot, expand-only autoscale)
    │   ├── gain_panel.py              (the one live gain-tuning panel, GainPanel)
    │   └── style.py                   (shared Okabe-Ito colors + side conventions)
    └── analysis/                      (every experiment script; figures → analysis/output/)
        ├── metrics.py                 (one metric definition: stats/print_stats/windowed_stats)
        ├── dashboard.py               (live 7-panel control-loop dashboard, both arms)
        ├── base_vs_error.py           (thesis success-criterion figure: base disp vs EE error)
        ├── diagnose.py                (pinned-scenario failure diagnosis, 01-08 figures)
        ├── bandwidth_sweep.py         (reactive-loop tracking bandwidth vs frequency)
        ├── validate_velocity.py       (ee_velocity vs finite-difference ground truth)
        └── fk_validation.py           (live direct-vs-FK comparison, right arm)
```

## 3. Module Status Table

| Module/File | Purpose | Status | Depends On | Used By | Tests | Confidence | Touch? |
| ----------- | ------- | ------ | ---------- | ------- | ----- | ---------- | ------ |
| `sim/scene.xml` | Torso + dual Gen3 scene, targets, contact excludes | Validated | `assets/kinova_gen3/gen3.xml` | `world.py` | Loads in all tests | High | Touch only if needed |
| `sim/assets/kinova_gen3/` | Vendored Gen3 MJCF (position-servo actuators) | Validated (kinematics + closed loop) | — | scene, `pin_fk.py` | Indirect via FK tests | High | Touch only if needed |
| `sim/world.py` | Model/data load, checked id lookups | Implemented | scene.xml | everything | Indirect (all tests import it) | High | Do not touch |
| `sim/targets.py` | Set/read target mocap poses | Implemented | `world.py` | desired_pos, servo | Indirect via `test_servo` wrapper tests | Medium-High | Touch only if needed |
| `sim/motion.py` | Scripted sinusoidal base motion; pure mechanism, off by default — each entry point passes its own scenario kwargs (zero amplitude = static, no mocap write) | Validated | world, transforms | main, plotting | `test_motion`: pose oracle, zero-guard, closed-loop rejection at 0.1 Hz | High | Scenario lives at the entry point (`BASE_SCENARIO` in main.py) |
| `controller/transforms.py` | Pure SE(3)/rotation math | Validated | numpy | kinematics, frames, desired_pos | `test_kinematics.RotationHelpersTest`, `test_transforms` (vs Pinocchio) | High | Do not touch |
| `controller/kinematics.py` | Analytical FK from MjModel constants | Validated, **demoted to test reference** | transforms | `test_pin_fk.py` only | `AnalyticalFKTest` vs MuJoCo, 50 cfg/arm, 1e-9 | High | Do not touch |
| `controller/pin_fk.py` | Pinocchio-backed `T_K_E(q)` — control path FK | Validated | pinocchio, `gen3.xml` | `frames.py` | `test_pin_fk` vs analytical FK, 1e-9 | High | Do not touch |
| `controller/frames.py` | World-frame EE pose + Jacobian | Validated (incl. moved torso) | pin_fk, transforms, world | desired_pos, servo, plotting | `WorldFrameEETest` (static + 50 random torso poses), `JacobianWorldTest` vs `mj_jacSite` | High | Touch only if needed |
| `controller/desired_pos.py` | Desired EE poses, resolved to world once | Implemented; `resolve_world` validated | frames, transforms, targets | `main.py`, plotting | `test_reference` (pure math) | Medium-High | Touch only if needed |
| `controller/servo.py` | The controller: P law + DLS (pure math) + MuJoCo plumbing (errors, setpoint integration, qdot limits) | Validated | frames, targets, world | main, plotting | `test_servo`: pure errors, DLS vs pinv, wrapper offsets, arm selection, `ClosedLoopConvergenceTest` | High | Touch only if needed |
| `main.py` | Viewer loop, closed loop, arm selection CLI | Implemented | desired_pos, servo, world | user | Loop body shared with `ClosedLoopConvergenceTest` via `apply_ctrl` | Medium-High | Grows with base motion |
| `plotting/live_plot.py` | Generic live plot, expand-only y-autoscale | Implemented | matplotlib | base_vs_error, fk_validation | None (visually exercised) | Medium | Do not touch |
| `plotting/gain_panel.py` | The one live gain-tuning panel (`GainPanel`, `on_change` hook) | Implemented | servo, matplotlib.widgets | main (tune), dashboard, base_vs_error | None (visually exercised) | Medium | Touch only if needed |
| `plotting/style.py` | Shared Okabe-Ito colors + side→color/linestyle maps | Implemented | — | dashboard, base_vs_error, diagnose, validate_velocity, bandwidth_sweep | None (constants only) | High | Touch only if needed |
| `analysis/metrics.py` | One metric definition: `stats`/`print_stats` (full-run) + `windowed_stats` | Implemented | numpy, world | base_vs_error, dashboard | Hand-verified: windowed_stats matches stats() on identical data | High | Touch only if needed |
| `analysis/dashboard.py` | Live 7-panel control-loop dashboard (per-axis error, e_rot, headroom, σ_min, margins) | Implemented | frames, servo, gain_panel, metrics | user | `test_dashboard.py`: pure per-tick helpers + DAMPING runtime guard | Medium-High | Touch only if needed |
| `analysis/base_vs_error.py` | Thesis success-criterion figure: base displacement vs. per-axis EE error | Implemented | frames, servo, live_plot, gain_panel, metrics | user | Visual + printed table only | Medium-High | Touch only if needed |
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

6. **Reactive controller (`controller/servo.py`)** —
   resolved-rate P law (v = Kp·e) with damped-least-squares inversion, driving
   position-servo setpoints as an integrator (ctrl += qdot·dt). Validated by
   pure-math tests (errors, DLS→pinv limit) and `ClosedLoopConvergenceTest`:
   both arms converge from home to feasible world targets under gravity in 3 s
   (< 5 mm, < 0.05 rad, error reduced ≥ 10×).

7. **Reference resolution (`controller/desired_pos.py::resolve_world`)** —
   torso-frame record → world pose, hand-computed compositions at 1e-12.
   Resolved **once** at startup; the controller consumes fixed world-frame
   mocap targets, so the task stays a world-frame hold by construction.

8. **Scene mounting (`sim/scene.xml`)** — arm mounts at torso-relative
   `(-0.16, ±0.10, 0.14)` with ±0.845708 rad roll; contact excludes for the
   mocap-welded base. Frozen unless the physical backpack geometry changes.

## 5. Incomplete Components

1. **Metrics + thesis plots** — no error logging, no RMSE/mean/peak, no
   publication figure path (`LivePlot` is live-monitoring only). Smallest next
   task: an experiment logger recording `(t, ref_pose, ee_pose)` arrays and a
   post-run summary.

2. **Known residual at the left task point** — the left `desired_pos` pose
   sits at joint_6's ctrl limit; the clip leaves a ~6 mm steady-state residual
   there by design, not a controller bug (see `ClosedLoopConvergenceTest`
   docstring). Revisit the task point or mount if it matters for the thesis
   scenario.

3. **Small hygiene** — `requirements.txt` lacks `matplotlib`.

## 6. Dependency Graph

```
sim/assets/kinova_gen3/gen3.xml (vendored, position-servo actuators)
├── sim/scene.xml (torso mocap + 2× <attach> + target mocap spheres)
│   └── sim/world.py (MjModel/MjData singletons + checked ids)
│       ├── sim/targets.py
│       ├── controller/frames.py  (pin_fk + transforms; pose + Jacobian)
│       ├── controller/kinematics.py  (transforms; TEST REFERENCE ONLY)
│       ├── controller/desired_pos.py  (frames + transforms + targets)
│       ├── controller/servo.py  (frames + targets; P law + DLS inside)
│       │   ├── main.py  (viewer closed loop, optional plotting/gain_panel.py)
│       │   ├── analysis/dashboard.py  (side from CLI arg; gain_panel + metrics)
│       │   └── analysis/base_vs_error.py  (both arms; gain_panel + metrics + live_plot)
│       └── analysis/fk_validation.py (frames, no controller)
│           └── plotting/live_plot.py  (matplotlib only, MuJoCo-free)
└── tests/  (test_kinematics, test_pin_fk, test_reference, test_transforms, test_servo, test_dashboard)
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
| Reactive controller | ✅ | ✅ (closed-loop convergence + 0.1 Hz rejection) | ⚠️ needs metrics at experiment conditions |
| Base motion scripting | ✅ | ✅ (`test_motion` oracle + rejection) | ✅ |
| Error metrics + plots | ✗ | ✗ | ✗ |

Blunt read: estimation/kinematics, the closed loop, and scripted base motion
are done and validated; what remains for the thesis is metrics/logging at the
experiment conditions. Measured: 50 mm sway at 0.1 Hz -> 15.5 mm peak EE error
(first-order theory predicts 15 mm); at 0.5 Hz the sensitivity w/sqrt(w^2+Kp^2)
is 0.84 — the reactive baseline barely rejects it, which is the thesis
motivation for prediction.

## 8. Next Work Queue

1. **Experiment logger + thesis metrics** — record `(t, ref, ee)` arrays;
   compute mean/RMSE/peak; save publication figures (offline path, not
   `LivePlot`). This is the phase's success criterion verbatim.
2. **Parameter sweep hooks** — base-motion amplitude/frequency as experiment
   parameters (already exposed as `set_torso_pose(t, **scenario)` kwargs;
   each entry point owns its scenario block, see §11); same metrics
   pipeline over ≥ 2 conditions.

Actuation decision (recorded): the reactive baseline commands **position-servo
setpoints via resolved-rate differential IK** (ctrl += qdot·dt). Torque control
would require swapping the vendored actuators for motors + gravity compensation;
revisit only if the NMPC comparison demands torque-level authority.

## 9. Do-Not-Redo List

- `controller/transforms.py` — pure math, tested to 1e-12. Done.
- `controller/kinematics.py` — analytical FK, deliberately demoted to test
  reference. Do not delete, do not port to Pinocchio, do not "unify" the two
  FK implementations — the redundancy is the verification strategy.
- `controller/pin_fk.py` — done and frozen.
- `controller/frames.py` composition (`T_W_T · T_T_K · T_K_E`) and Jacobian —
  validated including moved torso; matches the hardware sensing story.
- `controller/servo.py` control math — the P law and DLS inversion are the
  pure functions above the MuJoCo plumbing (`qdot_from_error` and friends).
- The circular-FK investigation — found, fixed, documented (2026-07-06 spec).
- `sim/scene.xml` mount geometry and contact excludes — working.
- `plotting/live_plot.py` — good enough for live monitoring; thesis figures are
  a separate offline path, not an upgrade of this.

## 10. Risks and Unknowns

1. **Mocap base motion has no consistent velocity.** Writing `mocap_pos` per
   step teleports the torso; mocap bodies carry no velocity state, so the arms
   feel base motion as position steps and any future feedforward must
   finite-difference it. Fine for the reactive baseline; a known modeling gap
   for anything predictive and for hardware (Vicon gives smooth trajectories).
2. **Actuator model mismatch.** Simulated Gen3 uses MuJoCo position servos with
   vendored gains; the real Gen3 exposes different low-level interfaces. Gains
   will not transfer — only the structure will.
3. **Torso is kinematic (mocap): no reaction forces.** The human doesn't feel
   the arms and arm inertia can't destabilize the base. Intended phase-1
   abstraction; on hardware the backpack dynamics are two-way. Don't over-tune
   to this idealization.
4. **Import-time side effects and relative paths.** `frames.py` builds the
   Pinocchio model and reads ids at import, with CWD-relative paths; everything
   must run from the repo root. Acceptable now; keep in mind for the C++ port.
5. **Two FK configs must stay in lockstep.** `pin_fk` loads `gen3.xml` directly;
   the scene attaches the same file. The analytical-FK cross-test is the
   tripwire — run tests after any `gen3.xml` edit.
6. **`desired_pos.apply()` and `targets.py` mocap writes** are exercised
   indirectly (wrapper tests, closed-loop test) but have no dedicated tests.

Resolved since the original audit: moved-torso coverage of `T_W_T` (now
tested), missing Jacobian (added + validated), empty controller (implemented +
validated), `torso_mocap_id` naming hazard (renamed `torso_body_id`), stale
README (refreshed), `reference.py` re-resolution hazard (targets are resolved
once into world-frame mocap bodies; the controller never re-resolves).

## 11. Extending the Sim

Five contracts for the ways this codebase is likely to grow (two-year
horizon, dozens of experiments).

1. **New trajectory shape** (circle, figure-eight, square wave): implement
   it as an `offset(t)`/`rate(t)` pair in `controller/transforms.py`,
   analytic derivatives of each other (like `sine_offset`/`sine_rate`), then
   swap the pair into `sim/target_motion.py`'s marked EXTENSION POINT —
   `target_pose_at` and `target_twist_at` together. An offset/rate pair that
   isn't an exact derivative breaks velocity validation silently; no test
   fails at the pose level.

2. **New disturbance** (chirp, step, recorded gait playback): a motion
   module is the triple `pose_at(t)` / `twist_at(t)` / `set_pose(t)`, with
   `twist_at` the analytic derivative of `pose_at`. `sim/motion.py` is the
   template. The entry point must call `set_pose` and `twist_at` as a
   matched pair, same scenario dict, same instant `t` — replacing only the
   pose write leaves the controller's twist feedforward silently wrong
   (main.py, analysis/dashboard.py, and analysis/base_vs_error.py all
   comment this at each call site; that's the rule to preserve).

3. **New controller** (the predictive one, eventually): a controller
   module is `init_ctrl()` + `apply_ctrl(dt, base_twist, arms)` +
   `pose_error(side)`; experiments select it by module name at their entry
   point. Keep the exact surface — a lesson already paid for. When
   `apply_ctrl`'s signature grew a required `base_twist`,
   `analysis/validate_velocity.py`'s two call sites were not updated and
   broke with a live `TypeError`, uncaught until this pass because the
   script has no test coverage. Before changing the surface, run
   `grep -rn "servo.apply_ctrl"` and update every hit.

4. **Another arm**: keyed by `world.SIDES` and the `f"{side}_..."` scene
   naming convention. Two hardcoded assumptions to revisit first:
   `controller/desired_pos.py`'s `POSES` dict hardcodes the side keys, and
   `controller/frames.py` shares one `gen3.xml` Pinocchio model across all
   sides — a different limb model per arm breaks that assumption.

5. **Disabling features**: base motion off = zero amplitudes in the entry
   point's scenario block (the default if omitted); target motion off =
   don't call `target_motion.set_target_pose` (off by default, no module
   levers); one arm only = the `main.py` CLI arg (`right`/`left`/`both`).
