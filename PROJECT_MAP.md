# Project Map

Audit date: 2026-07-07. Test suite at audit time: 14/14 pass (`python -m unittest discover tests`).
Working tree: one trivial uncommitted diff in `controller/kinematics.py` (trailing newline only).

## 1. System Goal

Hold a **world-frame end-effector pose** with a torso-mounted dual Kinova Gen3
(supernumerary robotic limbs) while the torso (the human) moves underneath it —
"chicken-head" stabilization. Current phase: the strongest possible **reactive**
baseline in MuJoCo, with error plots (mean, RMSE, peak) good enough for the
thesis. Predictive/MPC work comes later and is explicitly out of scope now.

The sensing/estimation story so far: EE pose is computed as
`T_W_E = T_W_T (torso, from mocap) · T_T_K (fixed mount) · T_K_E(q) (arm FK)`,
never read directly from MuJoCo — mirroring what will be available on hardware
(Vicon torso pose + joint encoders).

## 2. Current Pipeline

```
sim/scene.xml + sim/assets/kinova_gen3/gen3.xml   (MJCF: torso mocap + 2 attached Gen3)
└── sim/world.py            (MjModel/MjData singletons, cached body/site IDs)
    ├── sim/targets.py      (write/read target mocap spheres)
    ├── controller/frames.py
    │   ├── controller/pin_fk.py       (Pinocchio T_K_E(q) — CONTROL PATH)
    │   ├── controller/transforms.py   (pure SE(3) math)
    │   └── → world-frame EE pose  T_W_E = T_W_T · T_T_K · T_K_E(q)
    ├── controller/kinematics.py       (analytical FK — TEST REFERENCE only)
    ├── controller/reference.py        (desired EE pose records → targets)
    ├── controller/pd.py               (EMPTY — controller does not exist)
    ├── main.py                        (viewer loop: apply reference once, mj_step)
    └── plotting/
        ├── live_plot.py               (generic live time-series plot)
        └── fk_validation.py           (direct vs FK EE trace → plots/fk_validation.png)

MISSING LINKS in the pipeline:  error computation → controller → joint command
                                scripted base motion → metrics (RMSE/peak) → thesis plots
```

## 3. Module Status Table

| Module/File | Purpose | Status | Depends On | Used By | Tests | Confidence | Touch? |
| ----------- | ------- | ------ | ---------- | ------- | ----- | ---------- | ------ |
| `sim/scene.xml` | Torso + dual Gen3 scene, targets, contact excludes | Validated | `assets/kinova_gen3/gen3.xml` | `world.py` | Loads in all tests | High | Touch only if needed |
| `sim/assets/kinova_gen3/` | Vendored Gen3 MJCF (position-servo actuators) | Validated (kinematics); actuation unresolved | — | scene, `pin_fk.py` | Indirect via FK tests | High (FK) / Low (actuation) | Touch only if needed |
| `sim/world.py` | Model/data load, ID cache | Implemented | scene.xml | everything | Indirect (all tests import it) | High | Do not touch |
| `sim/targets.py` | Set/read target mocap bodies | Implemented | `world.py` | `reference.py` | None direct | Medium | Touch only if needed |
| `controller/transforms.py` | Pure SE(3)/rotation math | Validated | numpy | kinematics, frames, reference | `test_kinematics.RotationHelpersTest` (1e-12) | High | Do not touch |
| `controller/kinematics.py` | Analytical FK from MjModel constants | Validated, **demoted to test reference** | transforms | `test_pin_fk.py` only | `AnalyticalFKTest` vs MuJoCo, 50 cfg/arm, 1e-9 | High | Do not touch |
| `controller/pin_fk.py` | Pinocchio-backed `T_K_E(q)` — control path FK | Validated | pinocchio, `gen3.xml` | `frames.py` | `test_pin_fk` vs analytical FK, 20 cfg, 1e-9 | High | Do not touch |
| `controller/frames.py` | World-frame EE pose composition | Validated (static torso only) | pin_fk, transforms, world | reference, fk_validation, (future controller) | `WorldFrameEETest` (1e-9) — torso NOT moved in test | Medium-High | Touch only if needed |
| `controller/reference.py` | Desired EE pose records, resolve to world | Implemented; `resolve_world` validated | frames, transforms, targets | `main.py` | `test_reference` (pure math only; `apply()` untested) | Medium | Touch only if needed |
| `controller/pd.py` | Reactive controller | **Not started** (empty file) | — | nobody | None | — | Needs work |
| `main.py` | Viewer loop | Implemented (no control) | reference, world | user | None | Medium | Needs work (will grow the control loop) |
| `plotting/live_plot.py` | Generic live plot | Implemented | matplotlib | fk_validation | None (visually exercised) | Medium | Do not touch |
| `plotting/fk_validation.py` | Live direct-vs-FK comparison | Implemented | frames, world, live_plot | user | Visual only | Medium | Touch only if needed |
| `tests/test_kinematics.py` | FK + rotation + world-pose tests | Validated (passes) | — | — | is the test | High | Do not touch |
| `tests/test_pin_fk.py` | Pinocchio vs analytical FK | Validated (passes) | — | — | is the test | High | Do not touch |
| `tests/test_reference.py` | `resolve_world` pure-math tests | Validated (passes) | — | — | is the test | High | Do not touch |
| `README.md` | Setup + layout docs | Implemented, partially stale | — | — | — | Medium | Touch only if needed |
| `tasks/todo.md` | Historical task log | Stale (history, not current state) | — | — | — | — | Do not touch (append-only) |
| `docs/superpowers/` | FK design spec + plan (2026-07-06) | Done, matches code | — | — | — | High | Do not touch |
| `requirements.txt` | mujoco, numpy, pin | Incomplete (**matplotlib missing**) | — | — | — | Medium | Needs work (one line) |

## 4. Validated Components

1. **Rotation/transform math (`controller/transforms.py`)**
   - What: quat→R, Rodrigues axis-angle, RPY→R, pose↔transform.
   - How: hand-computed rotations + orthonormality/determinant/axis-invariance checks.
   - Test: `test_kinematics.RotationHelpersTest`. Tolerance: 1e-12.
   - Frozen: **yes**.

2. **Analytical FK (`controller/kinematics.py`)**
   - What: `extract_chain` + `fk` compute `T_K_E(q)` from MjModel constants only —
     genuinely independent of MuJoCo's pose outputs (the earlier circular version
     was found and replaced; see `docs/superpowers/specs/2026-07-06-analytical-fk-design.md`).
   - How: 50 random in-limit configurations per arm vs `mj_kinematics` site pose.
   - Test: `test_kinematics.AnalyticalFKTest` (both arms + 7-joint chain check). Tolerance: atol 1e-9 (actual errors ~1e-13 per session logs).
   - Frozen: **yes**. It exists to cross-check other FK implementations. Do not "improve" it.

3. **Pinocchio FK (`controller/pin_fk.py`)**
   - What: `pin_T_K_E(q)` from the arm-only MJCF; base_link-at-identity assertion guards frame validity.
   - How: 20 random configs vs analytical FK, position + geodesic rotation error.
     (Session logs also record a 50-config direct check vs MuJoCo at ~3e-15, but only the vs-analytical test is in the repo.)
   - Test: `test_pin_fk.PinFKTest`. Tolerance: 1e-9 m / 1e-9 rad.
   - Frozen: **yes** for FK. Will be *extended* (Jacobians) but not reworked.

4. **World-frame EE pose (`controller/frames.py`)**
   - What: `right/left_ee_pose()` = `T_W_T · T_T_K · T_K_E(q)` matches MuJoCo's
     world site pose, with the EE pose read from MuJoCo only on the comparison side.
   - How: randomize all joints, `mj_kinematics`, compare.
   - Test: `test_kinematics.WorldFrameEETest`. Tolerance: atol 1e-9.
   - Caveat: the test never moves the torso mocap body, so the `T_W_T` leg is
     exercised only at the nominal torso pose. Base-motion agreement has been
     seen only in the live plot (2026-07-05 session). **Not fully frozen** — add
     a moved-torso case to the test before treating it as such.

5. **Reference resolution (`controller/reference.py::resolve_world`)**
   - What: world/torso-frame record → world pose, pure math.
   - How: hand-computed compositions incl. yawed torso; error paths raise.
   - Test: `test_reference.TestResolveWorld`. Tolerance: 1e-12.
   - Frozen: yes for `resolve_world`. `apply()` (mocap write side) is untested.

6. **Scene mounting (`sim/scene.xml`)**
   - What: arm mounts at torso-relative `(-0.16, ±0.10, 0.14)` with ±0.845708 rad
     roll (world ≈ `(-0.16, ±0.10, 1.24)`), contact excludes for mocap-welded base.
   - How: implicitly by every FK test (mount transform is part of the validated chain).
   - Frozen: yes, unless the physical backpack geometry changes.

## 5. Incomplete Components

1. **Reactive controller (`controller/pd.py` — empty)**
   - Missing: everything. No error computation, no control law, no joint command path.
   - Depends on: frames (done), reference (done), an error definition, a Jacobian (not yet exposed), and an actuation decision (position servos vs torque).
   - Why it matters: it *is* the thesis baseline. Nothing downstream exists without it.
   - Smallest next task: a pose-error function `(pos_err, rot_err)` between a world reference and `frames.*_ee_pose()`, with a test (zero at coincidence, known offset recovered).

2. **Scripted base motion**
   - Missing: any code that moves the torso mocap body (no module does; `main.py` steps a static scene).
   - Depends on: nothing — `data.mocap_pos/quat` writes, same mechanism as `targets.py`.
   - Why it matters: the research question is performance *under base motion*; without it there is no disturbance and no experiment.
   - Smallest next task: `sim/base_motion.py` with a time-parameterized sinusoid `set_torso_pose(t)`.

3. **Actuation path decision + joint command**
   - Missing: the vendored Gen3 uses **position servo actuators** (`<position class="large_actuator">`). No decision recorded whether the reactive baseline commands servo setpoints (via differential IK) or torques (requires swapping actuators to motors + gravity compensation).
   - Depends on: controller design choice; affects C++ portability and the later NMPC comparison.
   - Why it matters: the whole comparison hinges on what quantity the controller commands; a servo-setpoint baseline and a torque baseline are different theses.
   - Smallest next task: write the decision down (one paragraph), then wire `data.ctrl` for one arm.

4. **Jacobian**
   - Missing: no Jacobian anywhere. Any Cartesian reactive law (resolved-rate, task-space PD) needs one.
   - Depends on: `pin_fk.py` (Pinocchio provides `computeFrameJacobian` on the already-built model).
   - Why it matters: maps task-space error to joint commands.
   - Smallest next task: `pin_frame_jacobian(q)` + finite-difference test against `pin_T_K_E`.

5. **Metrics + thesis plots**
   - Missing: no error logging, no RMSE/mean/peak computation, no publication figure path. `LivePlot` is live-monitoring only.
   - Depends on: controller + base motion existing.
   - Why it matters: success criterion is literally "error plots that can go straight into the thesis."
   - Smallest next task: an experiment logger that records `(t, ref_pose, ee_pose)` to arrays and a post-run summary (mean/RMSE/peak).

6. **Small hygiene items**
   - `requirements.txt` lacks `matplotlib`.
   - `world.torso_mocap_id` is a *body* id, not a mocap index — name is misleading (code is correct; `targets._mocap_index` shows the distinction).
   - README layout section is stale (`kinematics.py` description predates the frames/pin split).
   - One uncommitted cosmetic diff in `kinematics.py` (commit or revert).

## 6. Dependency Graph

```
sim/assets/kinova_gen3/gen3.xml (vendored, position-servo actuators)
├── sim/scene.xml (torso mocap + 2× <attach> + target mocap spheres)
│   └── sim/world.py (MjModel/MjData singletons + IDs)
│       ├── sim/targets.py
│       ├── controller/frames.py ──────────────┐
│       │   ├── controller/pin_fk.py  (also loads gen3.xml directly)
│       │   └── controller/transforms.py       │
│       ├── controller/kinematics.py  (transforms; TEST REFERENCE ONLY)
│       ├── controller/reference.py  (frames + transforms + targets)
│       │   └── main.py  (viewer loop)         │
│       └── plotting/fk_validation.py ◄────────┘
│           └── plotting/live_plot.py  (matplotlib only, MuJoCo-free)
├── tests/test_kinematics.py  (transforms + kinematics + frames vs MuJoCo)
├── tests/test_pin_fk.py      (pin_fk vs kinematics)
└── tests/test_reference.py   (reference.resolve_world, pure)

controller/pd.py  — empty, no edges yet
```

Note the deliberate redundancy: `kinematics.py` (analytical) and `pin_fk.py`
(Pinocchio) both compute `T_K_E`; the analytical one exists to catch regressions
in the Pinocchio path and must stay independent of it.

## 7. Research Readiness

| Subsystem | Implemented | Validated | Paper-ready |
| --- | --- | --- | --- |
| Scene / robot model | ✅ | ✅ (kinematically) | — (actuation model undecided) |
| Transform math | ✅ | ✅ | ✅ (nothing to publish, just trustworthy) |
| Arm FK (both impls) | ✅ | ✅ (1e-9 tol, dual-implementation cross-check) | ✅ as methodology |
| World-frame EE pose | ✅ | ⚠️ static torso only | ✗ |
| Reference/targets | ✅ | ✅ (pure part) | ✗ |
| Base motion scripting | ✗ | ✗ | ✗ |
| Reactive controller | ✗ (empty file) | ✗ | ✗ |
| Error metrics + plots | ✗ | ✗ | ✗ |

Blunt read: the **estimation/kinematics layer is genuinely done and well
validated**; the **control layer has not been started**. The project is at the
end of the foundation phase, not the middle of the control phase.

## 8. Next Work Queue

1. **Commit the working tree**
   - Objective: clean state (kinematics.py newline diff).
   - Prerequisite: none. Output: clean `git status`. Validation: `git status` empty, tests pass. Why: repo rule — every commit a working snapshot.

2. **Moved-torso case in `WorldFrameEETest`**
   - Objective: set torso `mocap_pos/quat` to a non-nominal pose in the test before comparing.
   - Prerequisite: none. Output: extended test. Validation: test passes with a translated + rotated torso. Why: closes the only untested leg (`T_W_T`) of the pose chain before the controller starts depending on it under base motion.

3. **Scripted base motion (`sim/base_motion.py`)**
   - Objective: `set_torso_pose(t)` writing a sinusoid to the torso mocap body; hook into `main.py`.
   - Prerequisite: task 2. Output: torso oscillates in the viewer; FK trace still matches in `fk_validation`. Validation: run `plotting/fk_validation.py` with base motion on — direct and FK traces must stay coincident while both move. Why: this is the disturbance the entire thesis measures against.

4. **Pose error function**
   - Objective: `(e_pos, e_rot)` between a world reference pose and `frames.*_ee_pose()`; rotation error as axis-angle/geodesic (reuse the `geodesic_angle` idea from `test_pin_fk`).
   - Prerequisite: none. Output: `controller/errors.py` (or similar) + unit test. Validation: zero at coincidence; a known injected offset is recovered exactly. Why: the controller and all thesis metrics consume this.

5. **Frame Jacobian from Pinocchio**
   - Objective: `pin_frame_jacobian(model, data, frame_id, q)` in `pin_fk.py`.
   - Prerequisite: none. Output: 6×7 Jacobian in a stated frame (world-aligned local, state it explicitly). Validation: finite-difference check against `pin_T_K_E` at random configs, ~1e-6 tol. Why: any Cartesian reactive law needs it; frame convention mistakes here are the classic silent killer.

6. **Record the actuation decision**
   - Objective: one written paragraph: reactive baseline commands position-servo setpoints via differential IK (or torques — decide), and why, and what changes for the hardware port.
   - Prerequisite: none (but do before task 7). Output: short note in README or docs/. Validation: n/a — it's a decision record. Why: prevents rework and makes the NMPC comparison well-posed later.

7. **Reactive controller v1, one arm, static torso**
   - Objective: fill `controller/pd.py`: task-space error → joint command per the task-6 decision; run in `main.py`'s loop with `reference.apply` semantics (world-frame hold).
   - Prerequisite: tasks 4–6. Output: right EE converges to and holds a world target with the torso still. Validation: steady-state position error below a stated bound (e.g. < 1 mm) in a headless run. Why: first closed loop; everything else is tuning and measurement.

8. **Closed loop under base motion**
   - Objective: run controller + `base_motion.py` together; EE holds world pose while torso oscillates.
   - Prerequisite: tasks 3, 7. Output: tracking-error time series. Validation: bounded error, no divergence, visibly better than open-loop (arms rigidly following the torso). Why: this is the actual experiment.

9. **Experiment logger + thesis metrics**
   - Objective: record `(t, ref, ee)` arrays per run; compute mean/RMSE/peak position (and orientation) error; save publication figures (not `LivePlot`).
   - Prerequisite: task 8. Output: `plots/` figures + printed metrics table. Validation: metrics recomputable from saved arrays; sanity-check RMSE against the plotted trace. Why: the phase's success criterion verbatim.

10. **Second arm + parameter sweep hooks**
    - Objective: enable the left arm in the same loop; make base-motion amplitude/frequency parameters of the experiment script.
    - Prerequisite: task 9. Output: repeatable multi-condition runs. Validation: same metrics pipeline over ≥2 conditions. Why: thesis needs curves over disturbance severity, not one anecdote.

## 9. Do-Not-Redo List

- `controller/transforms.py` — pure math, tested to 1e-12. Done.
- `controller/kinematics.py` — analytical FK. Done, *deliberately* demoted to test
  reference. Do not delete it, do not port it to Pinocchio, do not "unify" the two
  FK implementations — the redundancy is the verification strategy.
- `controller/pin_fk.py` — FK part is done. Extend (Jacobian) without touching `pin_T_K_E`.
- `controller/frames.py` composition structure (`T_W_T · T_T_K · T_K_E`) — correct
  and matches the hardware sensing story. Only the test coverage needs the moved-torso case.
- The circular-FK investigation — already found, fixed, and documented (2026-07-06 spec). Don't re-litigate.
- `sim/scene.xml` mount geometry and contact excludes — working; only change if physical backpack specs change.
- `plotting/live_plot.py` — good enough for live monitoring. Thesis figures should be a separate offline path, not an upgrade of this.
- Repo/venv/publish plumbing recorded in `tasks/todo.md` — historical, resolved.

## 10. Risks and Unknowns

1. **`T_W_T` under motion is test-uncovered.** All world-pose tests run at the
   nominal torso pose. A sign/convention bug in the torso leg would pass every
   current test and only appear once the base moves. (Mitigation = task 2.)
2. **Mocap base motion has no consistent velocity.** Writing `mocap_pos` per step
   teleports the torso; MuJoCo mocap bodies carry no velocity state, so the arms
   feel base motion as position steps, and any future controller term needing base
   *velocity/acceleration* (e.g. feedforward) must finite-difference it. Fine for
   the reactive baseline; a known modeling gap for anything predictive and for
   hardware, where Vicon gives smooth trajectories.
3. **Actuator model mismatch.** Simulated Gen3 uses MuJoCo position servos with
   vendored gains; the real Gen3 exposes different low-level interfaces. Whatever
   the baseline commands, calibrate expectations that gains will not transfer —
   only the structure will.
4. **Torso is kinematic (mocap): no reaction forces.** The human doesn't feel the
   arms, and arm inertia can't destabilize the base. That's the intended phase-1
   abstraction, but on hardware the backpack dynamics are two-way. Don't over-tune
   to this idealization.
5. **Import-time side effects and relative paths.** `frames.py` builds the
   Pinocchio model and reads IDs at import, with paths like
   `"sim/assets/kinova_gen3/gen3.xml"` relative to the CWD. Everything must run
   from the repo root; running from anywhere else fails at import. Also awkward
   for the C++ port's mental model — acceptable now, keep in mind.
6. **Two FK configs must stay in lockstep.** `pin_fk` loads `gen3.xml` directly;
   the scene attaches the same file. If the vendored model is ever edited (e.g.
   actuators swapped to motors), both paths change together — but the base-link
   identity assertion in `build_pin_model` only guards the root pose, not joint
   changes. The analytical-FK cross-test is the real tripwire; keep it in CI habit
   (run tests after any `gen3.xml` edit).
7. **`torso_mocap_id` naming.** It's a body id used to index `xpos` (correct), not
   a mocap index. Someone (including future-you) indexing `mocap_pos` with it will
   get a silent wrong-buffer bug. Rename or comment when next touching `world.py`.
8. **Default references are torso-frame records.** `reference.RIGHT/LEFT` use
   `frame: "torso"` and `apply()` resolves them against the torso *at call time*.
   The thesis task is a **world-frame hold** — the intended usage is resolve-once-
   then-hold-world. Nothing currently enforces that a controller doesn't re-resolve
   every tick (which would turn the task into torso-frame following and quietly
   destroy the experiment). Make the controller consume a fixed world pose.
9. **`reference.apply()` and `targets.py` have no tests** — thin mocap-write
   wrappers, low risk, but they sit on the experiment-critical path.
10. **`requirements.txt` incomplete** (`matplotlib` missing) and README layout
    stale — small, but both bite the "reproduce on a clean machine" moment, e.g.
    the eventual lab workstation.
