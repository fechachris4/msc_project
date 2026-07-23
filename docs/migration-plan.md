# Reactive controller migration plan

This plan preserves the current Python/MuJoCo reactive baseline while making
the control math portable to C++ and, later, Kinova BaseCyclic hardware.
Work pauses for review after every numbered step.

## Fixed rulings

1. All Cartesian controller math is world-aligned. Target-frame selection and
   transforms happen before the controller; controllers contain no frame
   conditionals.
2. The common backend operation is an atomic command/state exchange, with
   takeover and release around it. MuJoCo stepping and BaseCyclic sessions stay
   behind that boundary.
3. The presently supported common actuator command is joint position. A
   joint-velocity request is integrated before that boundary. True
   joint-velocity or torque actuation will use a separate explicit pipeline
   only after both backends genuinely support it.
4. Controller gains, software limits, target definitions, and nominal control
   period come from one TOML file read by Python and C++. The complete
   effective configuration is printed at startup and stamped into every log.
5. Pipelines remain explicit. A controller is not forced through inverse
   kinematics or position integration when its actual control flow bypasses
   either stage.
6. Live gain tuning will be removed in Step 2. Gains change by editing TOML and
   restarting the run; there is no second mutable gain source.

## Reset rule

Reset means reconstruction.

On startup, recovery, resume, or controller switching, the Runner obtains a
fresh plant state, reconstructs every stateful pipeline object, and seeds the
new objects from that state. This covers the position integrator, measurement
estimator history, and any stateful controller. Fault reports and the existing
telemetry buffer may survive reconstruction because they are run records, not
control state.

No component gets an event-specific reset branch. If reconstruction cannot
implement a required recovery safely, work stops for review before a special
case is introduced.

## Numerical parity gates

### Python refactor parity: Steps 2-4

The committed golden trace from Step 1 is the reference. Every numeric field
must match with:

- relative tolerance: `1e-12`
- absolute tolerance: `1e-12`
- integer, boolean, cycle, arm, and column identity: exact
- row count and ordering: exact

The absolute tolerance is required for quantities whose correct value is near
zero, where relative error is undefined or misleading. `1e-12` is far below
the controller's physical resolution but allows decimal CSV round trips and
harmless last-bit variation. These steps remain in Python and retain the same
NumPy/Pinocchio/MuJoCo environment, so a larger tolerance would risk hiding an
operation-order or frame change.

The same trace check remains a regression gate through Steps 5-6. Any intended
numeric change must be proposed separately rather than accepted by regenerating
the golden trace.

### Python/C++ parity: Step 8

Cross-language parity is evaluated by replaying the same recorded per-cycle
inputs through both pure pipelines, not by comparing two independently evolving
closed-loop simulations. This prevents a first-round floating-point difference
from being amplified by later plant states.

For controller and actuation floating-point outputs:

- relative tolerance: `1e-9`
- absolute tolerance: `1e-11`
- saturation decisions and other discrete fields: exact

The looser relative tolerance accounts for NumPy/Eigen decomposition and
pseudoinverse implementation differences while remaining negligible compared
with the controller's physical command scale.

On failure, the diagnostic procedure is:

1. Find the first failing cycle and arm.
2. Confirm the replay inputs first: time, `dt`, joint state, torso pose/twist,
   target pose, and mount calibration.
3. Compare the first divergent derived quantity in this order:
   world EE pose, world Jacobian, world EE twist, pose error, twist error,
   proportional twist, derivative twist, DLS task velocity, null-space
   velocity, raw joint velocity, clipped joint velocity, integrated position,
   lead/range clamp, final command.
4. Report the field, cycle, Python value, C++ value, absolute error, and
   relative error. Do not tune gains or widen tolerances until the first
   divergence is explained.

## Steps

### Step 1 — Freeze current behavior

- Add a deterministic harness around the current, unrefactored control path.
- Use fixed initial state, target script, torso-motion script, duration, and
  timestep.
- Record every cycle's plant/frame state, controller stages, and applied joint
  position command for both arms.
- Version the trace and its manifest.
- Prove a fresh rerun passes the trace comparator, then commit.

### Step 2 — Immutable TOML configuration

- Add the shared TOML schema and immutable validated Python configuration.
- Make gains, limits, targets, and nominal `dt` come from TOML.
- Print and log the effective configuration and raw-file hash.
- Remove the live gain panel and mutable gain globals.
- Pass the Python refactor parity gate.

### Step 3 — State and world-frame boundary

- Introduce fixed-shape plant, pose, twist, target, and controller-state types.
- Make frame transforms consume explicit state rather than MuJoCo globals.
- Keep `T_W_T` dynamic, `T_T_B` calibrated, and `T_B_E(q)` kinematic.
- Resolve world/base/torso targets into world pose and twist before control.
- Compare FK and frame outputs with MuJoCo ground truth.
- Pass the Python refactor parity gate.

### Step 4 — Pure controller and reconstruction-only state

- Extract the reactive pose law into one pure controller implementation.
- Extract velocity limiting and position integration into one stateful
  actuation object.
- Remove duplicated DLS calculations.
- Implement reset only by reconstructing and reseeding the pipeline.
- Pass the Python refactor parity gate.

### Step 5 — Runner and MuJoCo backend

- Add explicit Runner sequencing.
- Put MuJoCo model/data ownership, stepping, and lifecycle behind
  `takeover/exchange/release`.
- Keep the reactive pose-to-position pipeline explicit.
- Pass the golden-trace regression gate.

### Step 6 — Validate the Python reactive baseline

- Reproduce the last known-good behavior.
- Run the complete verification and thesis-metric path.
- Stamp effective configuration and provenance into artifacts.
- Pass the golden-trace regression gate.

### Step 7 — Mechanical C++ port with MuJoCo

- Port fixed types, transforms, kinematics, reactive controller, position
  integration, Runner, telemetry contract, TOML schema, and MuJoCo backend
  one-to-one.
- Do not add MPC or hardware behavior.

### Step 8 — Cross-language parity

- Replay identical golden inputs through Python and C++.
- Apply the cross-language tolerance and first-divergence procedure above.
- Run C++ MuJoCo closed-loop tests only after replay parity passes.

### Step 9 — Kinova BaseCyclic backend

- Add BaseCyclic sessions, cyclic exchange, servoing takeover/release, and
  synchronized Vicon state behind the established backend contract.
- Run the existing C++ controller without controller edits.
- Retain CSV replay for recorded hardware faults and exact input regression;
  use C++ MuJoCo as the primary closed-loop software gate.
