# Behaviour contract

What the C++ port must reproduce. Every clause is checkable, and every
clause was read out of the Python source rather than assumed. Clause ids are
referenced from the C++ tests.

## A. Units and frames

| id | Clause |
|----|--------|
| A1 | All internal quantities are SI: metres, radians, seconds. Millimetres appear only in printed output. |
| A2 | `T_A_B` = pose of B expressed in A. Frames: W world, T torso, B/K arm base, E end-effector site `pinch_site`. |
| A3 | Poses are `(position ∈ ℝ³ m, rotation ∈ SO(3))`. Rotation matrices, not quaternions, are the internal representation. |
| A4 | Quaternions are MuJoCo order `[w,x,y,z]` and occur only at MuJoCo boundaries (mocap writes, marker reads). |
| A5 | `rotation_from_rpy([r,p,y]) = Rz(y) · Ry(p) · Rx(r)`. |
| A6 | Jacobians are 6×7 with rows `[v; ω]`, expressed world-aligned about the EE point. |
| A7 | Controller inputs are world-frame only. Frame selection (`world`/`base`/`torso`) is resolved once per cycle at the Runner boundary. |

## B. Kinematics

| id | Clause |
|----|--------|
| B1 | `T_K_E(q)` comes from Pinocchio on `gen3.xml`, whose `base_link` is fixed at identity. Model build must assert that. |
| B2 | `T_W_E = T_W_T · T_T_B · T_B_E(q)`. MuJoCo's site pose is never used in the control path. |
| B3 | `J_W = blockdiag(R_W_B, R_W_B) · J_B`, with `J_B = getFrameJacobian(LOCAL_WORLD_ALIGNED)` and `R_W_B = R_W_T · R_T_B`. |
| B4 | EE twist: `v = v_torso + ω_torso × (p_E − p_T) + (J_W q̇)[0:3]`, `ω = ω_torso + (J_W q̇)[3:6]`. |
| B5 | Link-sphere world position: `p = p_B + R_W_B (p_frame + R_frame c)`; its point Jacobian is `R_W_B · (J_lin + J_ang × offset)` per sphere. |
| B6 | `MountCalibration` `T_T_B` is read from the MJCF `body_pos`/`body_quat` of `{side}_base_link`, not hard-coded. |

## C. Reference resolution

| id | Clause |
|----|--------|
| C1 | `world` frame → reference pose identity, reference twist zero. |
| C2 | `torso` frame → reference pose/twist are the torso's. |
| C3 | `base` frame → pose `T_W_T · T_T_B`; twist `v_torso + ω_torso × (p_B − p_T)`, angular `ω_torso`. |
| C4 | Resolved world target: `pose = frame_pose ∘ target_pose`; `v = v_f + ω_f × (R_f p_t) + R_f v_t`; `ω = ω_f + R_f ω_t`. |

## D. Controller mathematics

| id | Clause |
|----|--------|
| D1 | `e_pos = p_ref − p_meas`. |
| D2 | `e_rot = log3(R_ref · R_measᵀ)` using Pinocchio's `log3`. |
| D3 | `e_v = v_ref − v_meas`, `e_w = ω_ref − ω_meas`. |
| D4 | `p_twist = [Kp_pos e_pos ; Kp_rot e_rot]`; each 3-block is zeroed independently when `position_enabled` / `orientation_enabled` is false. |
| D5 | `d_twist = [Kd_pos e_v ; Kd_rot e_w]`, or all six zero when `velocity_enabled` is false. |
| D6 | `task_twist = p_twist + d_twist`. |
| D7 | `qdot_task = Jᵀ (J Jᵀ + λ² I₆)⁻¹ task_twist`, solved as a linear system (not by explicit inverse). |
| D8 | `qdot_null_obj = −(mask ⊙ k_null) ⊙ (q − q_mid)`, mask per joint. |
| D9 | `qdot_null = (I₇ − J⁺J) qdot_null_obj` with `J⁺` the **undamped** Moore–Penrose pseudo-inverse. |
| D10 | `qdot_raw = qdot_task + qdot_null`. |

## E. Human-safety geometry and filter

| id | Clause |
|----|--------|
| E1 | Envelope is a capped cylinder in the **torso** frame: axis torso-z, centre `center_xy_torso_m`, radius `radius_m`, extent `[z_min, z_max]`. |
| E2 | Signed distance uses the exterior corner rule: if both radial and vertical excesses are positive, distance is their hypotenuse and the gradient is the normalised blend; otherwise the larger of the two, with its own gradient. |
| E3 | On-axis degeneracy (radial < 1e-12) uses gradient `[1,0,0]`. |
| E4 | `signed_clearance = distance − sphere_radius − clearance_m`. |
| E5 | `distance_jacobian = gradient_torsoᵀ · J_point_torso` (1×7 per sphere). |
| E6 | A sphere is active iff `enabled AND NOT mount_exempt AND signed_clearance ≤ activation_distance_m`. |
| E7 | Constraint: `A qdot ≥ b`, `b = −recovery_gain (clearance − control_margin) − approach_damping · min(A q̇_meas, 0)`. |
| E8 | Resolution ladder exactly as in `01-architecture.md` §4: disabled → box-clip → no-active → already-feasible → repair(4) → OSQP + repair(16) → hold. |
| E9 | When disabled, the request is returned **unclipped**; joint-limit clipping is skipped entirely. |
| E10 | On hold, `qdot = clip(0, lower, upper)`: not literally zero, because the persistent command may need to catch up to its bounds. |
| E11 | `human_adjusted` uses `allclose(atol=tolerance, rtol=0)`; `limit_adjusted` uses exact array equality. |
| E12 | `limiting_points` are those with slack ≤ 10 × tolerance. |

## F. Actuation

| id | Clause |
|----|--------|
| F1 | The commanded position persists across cycles and is seeded from measured joint position at construction. |
| F2 | Order: speed clip → integrate → lead clamp about measured → range clamp to actuator `ctrlrange`. |
| F3 | `qdot_effective = (command_after − command_before)/dt`. |
| F4 | `velocity_bounds` returns `lower = max(−v_max, (max(q_meas − lead, q_lo) − command)/dt)` and the mirrored upper; it raises if the interval is empty. |
| F5 | Saturation flags are exact inequality comparisons against the pre-clip values. |

## G. Cylinder keep-out routing

| id | Clause |
|----|--------|
| G1 | Constants: `GEOMETRY_EPSILON = 1e-9`, `ARC_STEP_RAD = 15°`, `ROUTE_PADDING_M = 0.01`, `MAX_WAYPOINTS = 32`. |
| G2 | `obstacle_radius = radius + clearance`; `obstacle_z = [z_min − clearance, z_max + clearance]`; `route_radius = obstacle_radius / cos(ARC_STEP/2) + padding`. |
| G3 | A target inside the inflated cylinder is moved **radially out** to `route_radius`, never refused; `target_adjusted` is set. |
| G4 | `segment_intersects` clips the segment to the inflated height band, then tests closest approach to the axis in xy. |
| G5 | Candidates are ccw, cw, over; the winner is chosen by strict `<` on length in that order. |
| G6 | `Append` drops points within `GEOMETRY_EPSILON` of the previous and respects the 32-waypoint cap. |
| G7 | The follower advances while the current waypoint is within `waypoint_tolerance_m`, and replans only when the resolved **world** target changes. |
| G8 | Routing substitutes the target position only; orientation and twist are untouched. |

## H. Trajectory generation

| id | Clause |
|----|--------|
| H1 | Translation across waypoints is a **globally** minimum-integrated-jerk C² quintic spline; endpoint velocity and acceleration are zero, interior values solved jointly. |
| H2 | Orientation interpolates by principal-log SO(3) quintic per segment with zero angular rate and acceleration at every knot: `R(t) = R_left · exp(s(u) · log(R_leftᵀ R_right))`. |
| H3 | Minimum-jerk progress: `s = 10u³ − 15u⁴ + 6u⁵`, with the stated first and second time derivatives. |
| H4 | Circles are exact: `p = centre + r(cos θ · radial + sin θ · tangent)`, `θ = 2π·revolutions·s(u)`; `clockwise` negates the tangent; normal and start direction must be orthogonal. |
| H5 | Sampling before `t=0` returns the first pose at rest; at or after the end, the last pose at rest. |
| H6 | Outputs below 1e-14 in magnitude are zeroed (`_clean_small`). |
| H7 | A `TargetProgram` requires all segments in one frame, C² continuity at every boundary (atol 1e-10), and a final segment ending at rest. |
| H8 | `PeriodicTargetSource` requires the wrapped source to close continuously in frame, pose, twist and acceleration; sampling is `elapsed mod period`. |
| H9 | Automatic timing uses the closed forms `1.875·d/v_max` and `sqrt((10/√3)·d/a_max)`, taking the max over all applicable limits. |
| H10 | Explicit durations are validated against limits; derived durations are scaled up until they comply. |

## I. Configuration

| id | Clause |
|----|--------|
| I1 | TOML is validated **strictly**: exact key sets at every level, unknown keys are errors, missing keys are errors. |
| I2 | Cross-field rules: `kp_*` > 0 when the corresponding component is enabled; `kd_* < 1` when velocity feedback is enabled; `z_max > z_min` for both cylinders; cylinder bounds validated even when disabled. |
| I3 | Values are immutable after load. |
| I4 | The effective configuration is printed as sorted-key JSON, followed by `config_sha256=<sha256 of the raw file bytes>`. |
| I5 | `run.nominal_dt_s` overwrites `model.opt.timestep` after model load. |

## J. Timing and update order

| id | Clause |
|----|--------|
| J1 | Cycle order is exactly: sample → resolve → state → safety-evaluate → route → control+actuate → exchange. |
| J2 | The command computed from state *k* is applied, then the plant steps to *k+1*. One `mj_step` per cycle. |
| J3 | `dt` is `plant.nominal_dt_s`, carried on the state record. |
| J4 | Target-source time is elapsed since backend `takeover()`. |
| J5 | Viewer real-time pacing is presentation only and must not affect physics or control. |

## K. Initialisation

| id | Clause |
|----|--------|
| K1 | `takeover()` refreshes the torso mocap from the scripted driver at the current time, runs `mj_kinematics`, and returns the first `PlantState`. |
| K2 | The pipeline is constructed from that first state, seeding the position integrators from measured joint positions. |
| K3 | Cylinder followers are seeded at the measured EE position and their accepted-target cache is cleared. |
| K4 | Trajectory preparation resets the model, writes configured initial joints, zeroes velocities, and runs `mj_forward` before reading the state used to anchor the trajectory. |

## L. Experiment behaviour

| id | Clause |
|----|--------|
| L1 | Default torso amplitudes are zero → static torso; mocap is still written every step (home + 0). |
| L2 | Arm selection `right`/`left`/`both` selects which arms the **controller** drives; both arms are always simulated and always produce state. |
| L3 | Exactly one configured trajectory arm per run; a second is a hard error. |
| L4 | Target mocap bodies are display-only and are never read back as control input. |
| L5 | The golden-trace scenario disables human safety, keeps cylinder routing enabled, drives the torso with the specified amplitudes, and drives both EE targets sinusoidally. |

## M. Outputs

| id | Clause |
|----|--------|
| M1 | Startup prints the effective config JSON and `config_sha256=…`. |
| M2 | Every 250 steps: per arm `t`, `|e|` mm, per-axis error mm, Jacobian singular values; then route status; then human-safety status. |
| M3 | The trace CSV uses `.17g` formatting, `\n` line endings, and a stable column order. |
| M4 | Booleans in the trace serialise as `True`/`False` (Python `str(bool)`), because the comparison is textual for discrete columns. |

## Tolerances

The Python golden trace locks itself at `rtol = atol = 1e-12`. The port
adopts the same figure as its **target**, and reports the achieved figure
per column rather than relaxing it silently. Rationale and measured results
are in `docs/04-parity-report.md`.
