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
    ├── controller/frames.py
    │   ├── controller/pin_fk.py       (Pinocchio T_K_E(q) — CONTROL PATH)
    │   ├── controller/transforms.py   (pure SE(3) math)
    │   ├── → world-frame EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)
    │   └── → world-aligned 6x7 Jacobian (validated vs mj_jacSite)
    ├── controller/kinematics.py       (analytical FK — TEST REFERENCE only)
    ├── controller/desired_pos.py      (desired EE poses → world targets, once)
    ├── controller/servo.py            (controller: e → v = Kp*e → qdot via DLS → apply_ctrl)
    ├── main.py                        (viewer loop: closed loop via servo.apply_ctrl)
    └── plotting/
        ├── live_plot.py               (generic live time-series plot)
        ├── position_error.py          (shared live error plot, closed loop)
        └── fk_validation.py           (live direct-vs-FK comparison, right arm)

MISSING LINKS in the pipeline:  scripted base motion → metrics (RMSE/peak) → thesis plots
```

## 3. Module Status Table

| Module/File | Purpose | Status | Depends On | Used By | Tests | Confidence | Touch? |
| ----------- | ------- | ------ | ---------- | ------- | ----- | ---------- | ------ |
| `sim/scene.xml` | Torso + dual Gen3 scene, targets, contact excludes | Validated | `assets/kinova_gen3/gen3.xml` | `world.py` | Loads in all tests | High | Touch only if needed |
| `sim/assets/kinova_gen3/` | Vendored Gen3 MJCF (position-servo actuators) | Validated (kinematics + closed loop) | — | scene, `pin_fk.py` | Indirect via FK tests | High | Touch only if needed |
| `sim/world.py` | Model/data load, checked id lookups | Implemented | scene.xml | everything | Indirect (all tests import it) | High | Do not touch |
| `sim/targets.py` | Set/read target mocap poses | Implemented | `world.py` | desired_pos, servo | Indirect via `test_servo` wrapper tests | Medium-High | Touch only if needed |
| `controller/transforms.py` | Pure SE(3)/rotation math | Validated | numpy | kinematics, frames, desired_pos | `test_kinematics.RotationHelpersTest`, `test_transforms` (vs Pinocchio) | High | Do not touch |
| `controller/kinematics.py` | Analytical FK from MjModel constants | Validated, **demoted to test reference** | transforms | `test_pin_fk.py` only | `AnalyticalFKTest` vs MuJoCo, 50 cfg/arm, 1e-9 | High | Do not touch |
| `controller/pin_fk.py` | Pinocchio-backed `T_K_E(q)` — control path FK | Validated | pinocchio, `gen3.xml` | `frames.py` | `test_pin_fk` vs analytical FK, 1e-9 | High | Do not touch |
| `controller/frames.py` | World-frame EE pose + Jacobian | Validated (incl. moved torso) | pin_fk, transforms, world | desired_pos, servo, plotting | `WorldFrameEETest` (static + 50 random torso poses), `JacobianWorldTest` vs `mj_jacSite` | High | Touch only if needed |
| `controller/desired_pos.py` | Desired EE poses, resolved to world once | Implemented; `resolve_world` validated | frames, transforms, targets | `main.py`, plotting | `test_reference` (pure math) | Medium-High | Touch only if needed |
| `controller/servo.py` | The controller: P law + DLS (pure math) + MuJoCo plumbing (errors, setpoint integration, qdot limits) | Validated | frames, targets, world | main, plotting | `test_servo`: pure errors, DLS vs pinv, wrapper offsets, arm selection, `ClosedLoopConvergenceTest` | High | Touch only if needed |
| `main.py` | Viewer loop, closed loop, arm selection CLI | Implemented | desired_pos, servo, world | user | Loop body shared with `ClosedLoopConvergenceTest` via `apply_ctrl` | Medium-High | Grows with base motion |
| `plotting/live_plot.py` | Generic live plot | Implemented | matplotlib | position_error, fk_validation | None (visually exercised) | Medium | Do not touch |
| `plotting/position_error.py` | Live closed-loop error plots (side from CLI) | Implemented | servo, world, live_plot | user | Visual only | Medium | Touch only if needed |
| `plotting/fk_validation.py` | Live direct-vs-FK comparison | Implemented | frames, world, live_plot | user | Visual only | Medium | Touch only if needed |
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

1. **Scripted base motion** — nothing moves the torso mocap body yet; without
   it there is no disturbance and no experiment. Smallest next task:
   `sim/base_motion.py` with a time-parameterized `set_torso_pose(t)` sinusoid,
   hooked into `main.py`.

2. **Metrics + thesis plots** — no error logging, no RMSE/mean/peak, no
   publication figure path (`LivePlot` is live-monitoring only). Smallest next
   task: an experiment logger recording `(t, ref_pose, ee_pose)` arrays and a
   post-run summary.

3. **Known residual at the left task point** — `desired_pos.LEFT` sits at
   joint_6's ctrl limit; the clip leaves a ~6 mm steady-state residual there by
   design, not a controller bug (see `ClosedLoopConvergenceTest` docstring).
   Revisit the task point or mount if it matters for the thesis scenario.

4. **Small hygiene** — `requirements.txt` lacks `matplotlib`.

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
│       │   ├── main.py  (viewer closed loop)
│       │   └── plotting/position_error.py  (side from CLI arg)
│       └── plotting/fk_validation.py (frames, no controller)
│           └── plotting/live_plot.py  (matplotlib only, MuJoCo-free)
└── tests/  (test_kinematics, test_pin_fk, test_reference, test_transforms, test_servo)
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
| Reactive controller | ✅ | ✅ (closed-loop convergence, static base) | ⚠️ needs base-motion results |
| Base motion scripting | ✗ | ✗ | ✗ |
| Error metrics + plots | ✗ | ✗ | ✗ |

Blunt read: estimation/kinematics **and the static-base closed loop are done
and validated**; the experiment itself (base motion + metrics) has not started.

## 8. Next Work Queue

1. **Scripted base motion (`sim/base_motion.py`)** — `set_torso_pose(t)`
   sinusoid on the torso mocap body; hook into `main.py`. Validation: FK and
   direct traces stay coincident in `fk_validation` while both move; the
   closed loop keeps tracking.
2. **Closed loop under base motion** — the actual experiment: EE holds a world
   pose while the torso oscillates. Validation: bounded error, visibly better
   than uncontrolled arms.
3. **Experiment logger + thesis metrics** — record `(t, ref, ee)` arrays;
   compute mean/RMSE/peak; save publication figures (offline path, not
   `LivePlot`). This is the phase's success criterion verbatim.
4. **Parameter sweep hooks** — base-motion amplitude/frequency as experiment
   parameters; same metrics pipeline over ≥ 2 conditions.

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
