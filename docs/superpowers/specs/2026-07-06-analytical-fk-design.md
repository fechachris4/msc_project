# Analytical Forward Kinematics for the Gen3 Arms — Design

**Date:** 2026-07-06
**Status:** Approved for planning

## Problem

`right_ee_positions()` in `controller/kinematics.py` is circular: `T_KR_E` is
derived from MuJoCo's own world EE pose (`data.site_xpos`) and multiplied back
into the chain, so the composition algebraically cancels to MuJoCo's answer.
The 1e-16 "FK validation" error compares MuJoCo with itself and proves matrix
algebra, not forward kinematics. The planned torque NMPC needs `T_K_E(q)`
computed independently from joint angles.

## Goals

- An analytical FK `T_K_E(q)` for each Gen3 arm, whose only runtime input is
  the joint configuration. No `MjData` pose reads inside the FK.
- Works for both arms (`right_`, `left_`) from one implementation.
- Full pose output (4x4 homogeneous transform), matching the existing
  `transform_from_pose` / `pose_from_transform` conventions.
- A genuinely independent test: random configurations vs. MuJoCo.

## Non-goals

- Jacobians, dynamics, or CasADi/Pinocchio symbolics (planned later; this FK
  is the reference implementation they will be validated against).
- Arbitrary MuJoCo kinematic trees (free joints, sliders). Seven revolute
  joints per arm is the only supported shape.

## Design

### `KinematicChain` (new, in `controller/kinematics.py`)

Constructed once at load time per arm prefix. Reads **only compile-time
constants from `MjModel`** (never `MjData`):

- ordered body chain from `{prefix}base_link` to the body carrying the EE
  site (`{prefix}pinch_site`),
- each body's fixed parent offset: `model.body_pos`, `model.body_quat`,
- each hinge joint's axis (`model.jnt_axis`) and global qpos address
  (`model.jnt_qposadr`, resolved by joint name — after `<attach>`, the two
  arms' joints interleave in the global `qpos`, so indices are never assumed),
- the EE site's fixed offset on its parent body: `model.site_pos`,
  `model.site_quat`.

This satisfies the `tasks/lessons.md` rule: constants composed at load time
from a named source, no hand-transcribed magic numbers.

### `KinematicChain.fk(qpos) -> 4x4 T_K_E`

Plain NumPy loop: for each body in the chain,
`T = T · T_fixed(body) · Rot(axis_i, qpos[adr_i])`, then apply the site
offset. Explicit and simple — intended to be portable to CasADi later.

### World-frame composition

`right_ee_positions()` (plus a left twin) is rebuilt as
`T_W_E = T_W_T · T_T_K · T_K_E(q)`:

- `T_W_T`: torso mocap pose read from `data` — legitimate, it is a commanded
  input, not an FK output,
- `T_T_K`: fixed mount transform, extracted once from the model at load time,
- `T_K_E(q)`: the analytical FK above.

The circular read of the EE world pose (old line 63) is deleted.

### Testing

In `tests/test_kinematics.py`, for each arm:

- seeded RNG samples ~50 configurations within `model.jnt_range`,
- write into `data.qpos`, call `mj_kinematics`,
- assert analytical pose matches `data.site_xpos` / `site_xmat` to ~1e-10
  (independent computations accumulate real floating-point differences;
  1e-16 agreement would itself be a circularity smell).

`plotting/fk_validation.py` is updated to plot the analytical FK against
MuJoCo, so the plot shows genuine independent-FK error.

## Affected files

- `controller/kinematics.py` — add `KinematicChain`, rebuild world-frame
  functions on it, remove the circular derivation.
- `tests/test_kinematics.py` — replace the self-comparison test with random
  q-sweep validation for both arms.
- `plotting/fk_validation.py` — switch to the analytical chain.
