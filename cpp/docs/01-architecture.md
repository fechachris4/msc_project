# Discovery: how the Python simulation works

Written before any C++ was authored. Everything here was read out of the
Python sources; nothing is inferred from the C++ port.

Source of truth: `/Users/christian/Projects/Code/msc_project` (Python).
Baseline verified at discovery time: `python -m tests.golden_trace --check`
→ `PASS: 500 rows match (rtol=1e-12, atol=1e-12)`.

## 1. What the simulation is

A MuJoCo simulation of a torso-mounted **dual Kinova Gen3** pair
(supernumerary robotic limbs). The torso is a **mocap body** representing the
human wearer. The task is "chicken-head" stabilisation: hold a **world-frame
end-effector pose** while the torso moves underneath.

The defining architectural constraint: **the EE pose is never read from
MuJoCo in the control path.** It is composed as

    T_W_E = T_W_T · T_T_K · T_K_E(q)

from the torso pose (mocap today, Vicon on hardware), a fixed mount
calibration, and arm FK from joint encoders. MuJoCo's own site pose is used
only for ground-truth tests. Any port must preserve this, or it silently
stops mirroring the hardware sensing story.

## 2. Entry point and startup order

Import-time side effects matter here, because they fix the order.

1. **`runtime_config.py`** (module import) reads `config/control.toml` via
   `tomllib`, validates it *strictly* (exact key sets, ranges,
   cross-field rules), and exposes a frozen `CONFIG` plus `source_sha256`.
2. **`sim/world.py`** (module import) constructs one default
   `MujocoBackend()`:
   - loads `sim/scene.xml`, which `<attach>`es two copies of
     `assets/kinova_gen3/gen3.xml` under the mocap `torso` body with the
     prefixes `right_` / `left_`;
   - **overrides** `model.opt.timestep` with `run.nominal_dt_s` (0.002 s);
   - creates `MjData`, runs `mj_forward`;
   - resolves named ids (EE sites, torso body, arm base bodies, target
     bodies, 7 actuators/joints per side) and fails loudly if any is missing;
   - builds `MountCalibration`: `T_T_B` per side, taken from
     `model.body_pos` / `body_quat` of `{side}_base_link`;
   - builds `DualArmPipelineSetup`: joint-centering midpoints from
     `jnt_range` (only for *limited* joints; unlimited joints get midpoint 0
     and are masked off), and actuation limits from TOML velocity limits +
     `position_lead_rad` + the **actuator `ctrlrange`** as position bounds.
3. **`controller/frames.py`** (module import) builds a **Pinocchio** model
   from `gen3.xml` *alone*, where `base_link` sits at the origin, so
   `oMf(pinch_site)` is directly `T_K_E`. It also precomputes the link-sphere
   frame groupings used by the safety filter.
4. **`main.py`** then: parses `[right|left|both]` and `--trajectory-plot`,
   prints the effective config + sha256, builds the static framed targets
   from TOML, installs the scripted torso driver, optionally materialises a
   configured trajectory, constructs `ReactivePositionRunner`, and starts it.

### Trajectory initialisation (only when `[targets.<side>.trajectory]` exists)

`sim/target_trajectory.prepare_target_trajectory` runs **before** the loop:
releases and resets the backend, writes
`[simulation.initial_joint_position_rad]` into `qpos`, zeroes `qvel`,
`mj_forward`, reads a `PlantState`, resolves the trajectory start pose
(`"measured"` = current EE pose, or `"configured_target"` = the static TOML
target), expresses it in the trajectory's declared frame, and compiles the
segment list into a `TargetProgram`. If `loop = true` the program is wrapped
in a `PeriodicTargetSource`. Exactly **one** trajectory arm per run is
allowed; the other arm holds its static target.

## 3. The control cycle, in order

`ReactivePositionRunner.cycle()` owns the ordering. Per cycle:

1. `target_elapsed = plant.sample_time_s − target_time_origin_s`
   (the origin is captured at `takeover()`, so trajectory time starts at
   backend takeover, not at wall-clock zero).
2. **Sample the target source once** → `DualArmFramedTargets`.
3. **`frames.resolve_targets_world`**: convert world/base/torso framed
   targets into world pose *and* world twist. Torso-carried frames
   contribute `v + ω × r`. After this point the controller sees no frame
   selector.
4. **`frames.controller_states`**: per arm:
   `pin.computeJointJacobians`, `pin.updateFramePlacements`;
   `T_W_E = T_W_T · T_T_B · T_B_E`;
   world Jacobian `J_W = blockdiag(R_W_B, R_W_B) · J_B` where
   `J_B` is `LOCAL_WORLD_ALIGNED`;
   EE twist `= torso_v + ω × (p_E − p_T) + J_W q̇` (linear) and
   `ω + (J_W q̇)[3:]` (angular);
   plus every link-sphere world position and its 3×7 point Jacobian.
5. **`human_safety.evaluate_dual_arm`**: express sphere centres and their
   Jacobians in the **torso** frame, compute signed clearance to a capped
   cylinder, build the distance Jacobian, and mark constraints active.
6. **`_route_targets`**: cylinder keep-out routing substitutes the active
   waypoint for the target **position only**; orientation and twist pass
   through untouched. A route is replanned whenever the *resolved world*
   target moves (so a torso-frame target replans as the torso moves).
7. **`pipeline.step`**: per selected arm, in this order:
   pose error → twist error → PD → DLS → null-space → safety projection →
   clip and integrate the persistent position command.
8. **`backend.exchange(command)`**: write `data.ctrl`, `mj_step`, refresh
   the torso mocap from the scripted driver, `mj_kinematics`, read the next
   `PlantState`.

Note the boundary: the command computed from state *k* is applied and then
the plant is stepped to *k+1*. One command per step, no sub-stepping.

## 4. Controller mathematics (`reactive_controller.py`)

The file is deliberately ordered as an executable equation sheet.

1. **Pose error** (world, reference − actual):
   `e_pos = p_ref − p`;  `e_rot = log3(R_ref · Rᵀ)`.
2. **Twist error**: `e_v = v_ref − v`;  `e_w = ω_ref − ω`.
3. **Task twist**:
   `p_twist = [Kp_pos·e_pos ; Kp_rot·e_rot]` (each half zeroed if its
   component is disabled);
   `d_twist = [Kd_pos·e_v ; Kd_rot·e_w]` (all six zeroed if velocity
   feedback is disabled);
   `xdot = p_twist + d_twist`.
4. **Damped least squares**:
   `qdot_task = Jᵀ (J Jᵀ + λ² I₆)⁻¹ xdot`.
5. **Null-space centering**:
   `qdot_null_obj = −k_null · mask · (q − q_mid)`.
6. **Null-space projection**:
   `qdot_null = (I₇ − J⁺ J) qdot_null_obj`, with `J⁺` the *undamped*
   pseudo-inverse. `qdot_raw = qdot_task + qdot_null`.
7. **Safety projection**: see below.

`k_null` is `centering.enabled * null_gain_s_inv`, i.e. a **per-joint
boolean mask times a scalar**, so unlimited joints (1/3/5/7 on the Gen3) are
not centred at all.

### Safety projection (equation 7)

Control-barrier form: for every *active* sphere,

    distance_jacobian · qdot ≥ −recovery_gain · (clearance − control_margin)
                               − approach_damping · min(measured_rate, 0)

The resolution ladder is important and must be preserved exactly:

1. If `human_safety.enabled` is false → return `qdot_raw` unchanged,
   reason `"disabled"`. **No joint-limit clipping happens here.**
2. Else clip the request into `[lower, upper]` (the exact clamp-free
   velocity interval from the integrator).
3. If no active constraints → return the bounded request, reason `"clear"`.
4. If the bounded request already satisfies every constraint within
   tolerance → return it (reason `"clear"` or `"joint_limit_filtered"`).
5. Else run `_repair_constraint_feasibility`: a Gauss–Seidel-style
   projection onto each violated half-space, re-clipped to the box, up to
   4 sweeps.
6. Only if that still violates → an **OSQP** solve on row-normalised
   constraints, then up to 16 more repair sweeps.
7. If still infeasible → **hold**: `qdot = clip(0, lower, upper)` and report
   `stopped`, with reason `"unsafe_initial_state_hold"` (some clearance was
   already negative) or `"constraint_projection_failed_hold"`.

So OSQP is a fallback of a fallback. Whether it is ever reached in a given
scenario is an empirical question, and the port instruments it.

### Actuation (`position_actuation.py`)

Persistent commanded position, seeded from the **measured** joint positions
at pipeline construction, then per cycle:

    qdot_clipped = clip(qdot, ±v_max)
    integrated   = command + qdot_clipped · dt
    lead_limited = clip(integrated, q_meas ± lead_rad)
    command      = clip(lead_limited, ctrl_lower, ctrl_upper)
    qdot_effective = (command_after − command_before) / dt

`velocity_bounds()` inverts that chain to give the exact velocity interval
that avoids every downstream clamp; this is what the safety filter is
handed as its box. Reset means *reconstruct the object*, never mutate.

## 5. Frames, units, conventions

- `T_A_B` is the pose of B expressed in A. W = world, T = torso,
  K/B = Kinova arm base, E = end-effector (`pinch_site`).
- Poses travel as `(position (3,) m, rotation 3×3)`. Quaternions are MuJoCo
  order `[w, x, y, z]` and appear only at MuJoCo boundaries.
- `rotation_from_rpy` composes **Rz(yaw) · Ry(pitch) · Rx(roll)**.
- Everything internal is SI (metres, radians, seconds). Millimetres appear
  only in prints and plots.
- Jacobians are 6×7, rows `[v (m/s); ω (rad/s)]`, world-aligned.
- Controller math is entirely world-frame; frame selection is resolved once
  per cycle at the Runner boundary.

## 6. Timing

- `nominal_dt_s = 0.002` (500 Hz) from TOML, and it **overwrites** the
  MJCF timestep.
- The integrator is `implicitfast`.
- `dt` used by the controller is `plant.nominal_dt_s`, read from the state
  record, not a separate clock.
- Target-source time is elapsed since backend takeover.
- The viewer loop sleeps to pace against `model.opt.timestep`; this is
  presentation only and does not enter the physics or the control math.

## 7. Scripted disturbances

- **Torso** (`sim/motion.py`): `pos = home + A·sin(2πft)`,
  `rpy = 0 + A_rot·sin(2πf_rot t)`, with the analytic twist supplied
  separately because mocap writes carry no velocity. Module-level default
  amplitudes are **all zero**, so the default scenario has a static torso.
  The rpy element-wise sum is only valid because the torso home rotation is
  identity, asserted at import.
- **EE target** (`sim/target_motion.py`): sinusoid about a captured home
  pose, off by default, used by the golden trace and bandwidth sweeps.
  Composed as `R_home · R_rpy(δ)` (matrix composition, not rpy addition).

## 8. Outputs

- **Console**: effective config JSON + sha256 at startup; then every 250
  steps, per-arm `|e|` in mm, per-axis error, and Jacobian singular values;
  cylinder route status; human-safety clearance/reason.
- **Viewer**: MuJoCo passive viewer with `user_scn` overlays for the
  keep-out cylinder, routes, safety envelope and link spheres. Target mocap
  spheres are display markers, never controller inputs.
- **Golden trace** (`tests/golden_trace.py`): 250 cycles × 2 arms → CSV at
  `.17g`, plus a JSON manifest with a SHA-256 of the CSV. This is the
  behaviour lock and the natural parity oracle for a port.
- **Analysis scripts** (`analysis/`): matplotlib figures and stamped
  baseline artifacts.

## 9. Things a port can get wrong

Recorded during discovery, each verified against the source:

1. `human_safety.enabled = false` bypasses joint-limit clipping too, not
   just the human constraints. Getting this "tidier" changes the trace.
2. `centering.enabled` is a per-joint mask multiplied by the scalar gain,
   not a global on/off.
3. `J⁺` in the null-space projector is **undamped** while the task solve is
   damped. That asymmetry is deliberate.
4. The cylinder route replans on *resolved world* target change, which
   includes motion induced purely by the torso.
5. Route waypoints replace the target **position** only; the requested
   orientation is used at intermediate waypoints.
6. The position integrator is seeded from measured position at construction,
   so "reset" must mean reconstruction.
7. `model.opt.timestep` is overwritten from TOML after model load.
8. Target markers are display-only; the Runner resolves retained records.
9. `_clean_small` zeroes trajectory outputs below 1e-14, a real numerical
   behaviour, not cosmetic.
10. Ties between cylinder route candidates resolve by strict `<` in the
    order ccw, cw, over.
