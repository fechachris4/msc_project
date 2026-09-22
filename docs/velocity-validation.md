# EE world velocity: derivation and validation

**Claim.** `frames.ee_velocity(side, base_twist)` returns the true
world-frame end-effector twist under scripted base motion, verified to
0.5 mm/s / 0.03 deg/s against a MuJoCo ground truth that is independent
of the production math. MuJoCo's own velocity readout is wrong by up to
315 mm/s / 15 deg/s in the same scenario and must not be used.

Produced by `python -m analysis.validate_velocity`
(figures `analysis/output/velocity_[123]_*.png`). Companion unit tests:
`tests/test_velocity.py`.

## 1. Why this exists

The torso is a mocap body teleported each step (`sim/motion.py`), so it
has no velocity state. Everything MuJoCo derives from `qvel` —
`mj_objectVelocity`, `data.cvel`, `framelinvel` sensors — treats the
base as bolted down and reports only the arm-relative part of the EE
velocity. Figure 3 shows the consequence: under base sway the direct
readout is roughly in antiphase with the truth (it sees only the arm's
compensating motion). The EE world velocity therefore has to be
composed explicitly, exactly as the EE pose already is.

## 2. Derivation

The pose pipeline computes `T_W_E = T_W_T(t) · T_T_K · T_K_E(q)`
(`controller/frames.py`). Differentiate with `T_T_K` constant. Writing
`p_TE` for the EE position relative to the torso expressed in world
(`p_E − p_T = R_W_T p̄_TE` with `p̄_TE` torso-frame):

```
p_E(t)  = p_T(t) + R_W_T(t) p̄_TE(q)
ṗ_E     = ṗ_T + Ṙ_W_T p̄_TE + R_W_T ṗ̄_TE
        = v_T + ω_T × (p_E − p_T) + R_W_T ṗ̄_TE
R_W_E   = R_W_T R̄_TE(q)   ⇒   ω_E = ω_T + R_W_T ω̄_TE
```

The arm-relative rates are the existing world-aligned Jacobian applied
forward: `J_world = R_W_K J_K` with `R_W_K = R_W_T R_T_K`
(`frames.jacobian_world`), and since `T_T_K` is constant,
`R_W_T [ṗ̄_TE; ω̄_TE] = J_world(q) q̇`. Hence the production formula:

```
v_E = v_T + ω_T × (p_E − p_T) + (J_world q̇)_lin      [m/s, world]
ω_E = ω_T + (J_world q̇)_ang                          [rad/s, world]
```

The base twist comes from the analytic derivative of the scripted
sinusoid (`motion.torso_twist_at`):

```
v_T   = A_lin · 2πf · cos(2πf t)
ṙpy   = A_rot · 2πf · cos(2πf t)
ω_T   = E(rpy) ṙpy,   E = [ Rz Ry ex | Rz ey | ez ]   (columns)
```

where `E` maps rpy rates to world angular velocity for the repo's
`R = Rz(yaw) Ry(pitch) Rx(roll)` convention
(`transforms.angular_velocity_from_rpy_rates`): each rate spins about
its own current axis; roll's axis is `Rz Ry ex`, so roll itself never
enters `E`.

## 3. Architecture

Each piece lives next to the code it differentiates — no new module:

| Piece | Home |
|---|---|
| `angular_velocity_from_rpy_rates` | `controller/transforms.py` (pure math, beside `rotation_from_rpy`) |
| `torso_twist_at` | `sim/motion.py` (beside `torso_pose_at`, same levers) |
| `arm_dof_adrs` / `dof_adrs`, `ee_velocity` | `controller/frames.py` (beside `ee_pose`, `jacobian_world`) |

`base_twist` is a **required argument** of `ee_velocity`. `frames`
deliberately does not import `sim.motion`: the twist source is
swappable (scripted derivative in sim, Vicon estimate on hardware,
zeros for a stationary base), and a forgotten base twist fails loudly
instead of defaulting to a silently wrong zero. Nothing in the control
law consumes the velocity yet; this is estimation infrastructure for
the base-motion feedforward work.

## 4. Validation methodology

Three legs, each independent of the production math it checks:

1. **Analytic-twist oracle** (pure math, no sim):
   `torso_twist_at` vs central finite differences of `torso_pose_at`
   (angular via `log3(R(t+h) R(t−h)ᵀ)/2h`), two scenarios including
   30–40° amplitudes to exercise the nonlinearity of `E`. Agreement
   ≤ 1e-6 m/s, rad/s.
2. **Arm term vs Pinocchio** (independent library path):
   `ee_velocity` with zero base twist vs
   `R_W_K · getFrameVelocity(LOCAL_WORLD_ALIGNED)` at 20 random
   configurations, joint rates, and torso poses. Agreement ≤ 1e-9.
   (`jacobian_world` itself is separately validated against MuJoCo's
   `mj_jacSite` in `test_kinematics.py`.)
3. **End-to-end ground truth** (MuJoCo, not Pinocchio): closed-loop
   rollout, 5 s of pinned sway (lin [50, 20, 10] mm @ 0.5 Hz, rot
   [5, 3, 8]° @ 0.3 Hz — rotation kept below the ~15.6° roll contact
   regime from `docs/diagnosis.md`), composed velocity vs central
   finite differences of `measured_ee_pose` (`site_xpos`, which does
   include the mocap teleports). This is the only leg that exercises
   the `ω×r` transport term. Per sample the mocap is rewritten and
   `mj_kinematics` re-run so pose, `qvel`, and analytic twist refer to
   the same instant.

## 5. Quantitative results

Composed vs FD-of-measured ground truth, per axis
(`python -m analysis.validate_velocity`, dt = 2 ms):

| arm | signal | RMSE x/y/z | max x/y/z | peak signal |
|---|---|---|---|---|
| right | v [mm/s] | 0.25 / 0.07 / 0.10 | 0.50 / 0.13 / 0.19 | 326 |
| right | ω [deg/s] | 0.01 / 0.01 / 0.02 | 0.02 / 0.02 / 0.03 | ~19 |
| left | v [mm/s] | 0.21 / 0.09 / 0.05 | 0.38 / 0.15 / 0.13 | 207 |
| left | ω [deg/s] | 0.01 / 0.01 / 0.02 | 0.03 / 0.02 / 0.03 | ~19 |

Worst case 0.15 % of peak — the O(dt·q̈) skew of comparing a
continuous-time estimate against a discrete integrator's trajectory,
not a math error (it shrinks with dt). MuJoCo's direct readout against
the same ground truth: RMSE up to 156 mm/s, max 315 mm/s and 15 deg/s
(right arm) — the entire base contribution is missing.

Unit tests pin these margins: leg 3 asserts max error < 2 mm/s
(mrad/s), ~4× the measured worst case, plus a 50 mm/s activity floor
proving the disturbance engaged. 30/30 tests pass.

## 6. Assumptions

- **Kinematic base**: the torso pose is prescribed (mocap); no dynamic
  coupling back from the arms. Holds in this phase by construction.
- **Exact scenario knowledge**: the analytic base twist assumes the
  torso follows `torso_pose_at` exactly, which mocap writes guarantee.
  On hardware this becomes a Vicon-derived estimate.
- **`T_T_K` constant**: the mount transform is rigid (model constant).
- **Measured `q̇`**: `data.qvel` is exact in sim; encoder-derived rates
  on the real arm are noisier.
- **RPY convention**: `E` is specific to `Rz Ry Rx`; it is validated
  against the same `rotation_from_rpy` the rest of the repo uses.

## 7. Limitations

- No sensor noise or latency: on hardware, Vicon base twists arrive
  late and filtered, and differentiated encoder signals amplify noise.
  The composition is unchanged; the twist *source* is what degrades.
- The FD ground truth is a sim-only validation tool (needs clean poses
  at 500 Hz); it is not a runtime estimator and never enters the
  controller.
- `E(rpy)` is singular at pitch = ±90° (gimbal lock of the rpy chart).
  Irrelevant at the scripted amplitudes; a hardware base-twist source
  would come as an angular velocity directly and never pass through `E`.
- Nothing consumes the velocity yet; closed-loop benefit (feedforward)
  is future work by design.
