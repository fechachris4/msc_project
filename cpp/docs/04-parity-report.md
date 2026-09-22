# Parity report

Sections 1, 2 and 4 are reproduced by `bash cpp/tools/verify_parity.sh` on
the current config. Section 3 is the last result before the Python trace
format changed; the script now skips it.

Environment: macOS (Darwin 27), Apple clang 21, CMake 4.3.4.
Python 3.14 with numpy 2.4.4 (Accelerate BLAS/LAPACK), mujoco 3.10.0,
pinocchio 4.0.0, scipy 1.17.1, osqp 1.1.1.
The C++ links the **same** MuJoCo and Pinocchio binaries out of the venv.

## Tolerance

The Python golden trace locks itself at `rtol = atol = 1e-12`
(`tests/golden_trace.py`), and the port was written against that bar. The
golden-trace comparison now uses 1e-11: with the tuned gains (Kp = 32)
joint-rate commands are about 16x larger than when the port was written, so
the same one-ULP differences grow to a worst case of 2.4e-12. The trajectory
comparison stays at 1e-12.

`math.isclose(a, b, rel_tol=tol, abs_tol=tol)` is the predicate, matching
the Python harness's `np.isclose` usage.

## Results

| # | Comparison | Scope | Verdict | Worst absolute difference |
|---|---|---|---|---|
| 1 | Effective configuration | full JSON + sha256 | **byte-identical** | 0 |
| 2 | Golden trace | 500 rows x 210 cols = 105,000 fields | **PASS** @1e-11 | 2.416e-12 (`qdot_raw_5`) |
| 3 | Headless trace (last run before the format change; now skipped) | 4,000 rows x 59 cols = 236,000 fields | **PASS** @1e-12 | 4.441e-13 (`qdot_safety_filtered_1`) |
| 4 | Trajectory sampling | 401 samples x 6 quantities | **PASS** @1e-12 | 1.127e-14 (linear acceleration) |
| 5 | C++ unit tests | 10 suites | **PASS** | - |

### 1. Configuration

`srl_print_config` and `runtime_config.print_effective_config` produce
byte-identical output, including CPython's `repr()` float formatting
(`0.002`, `1.1`, `1e-06`, `1.3892820845874863`) and the SHA-256 of the raw
TOML bytes.

### 2. Golden trace (human safety disabled, routing enabled)

Reuses the committed `tests/golden/reactive_current.csv` as the reference, so
this compares against a *frozen artifact*, not a fresh Python run.

- 24 of 187 numeric columns are **bit-identical**, including
  `sample_time_s`, `dt_s` and all three `torso_position_m` components.
- On the pre-tuning trace, divergence first appeared at **cycle 0** in `J`
  (5.55e-16) and `ee_position_m` (4.44e-16), one ULP, and `q` (the MuJoCo
  joint state) was bit-identical until **cycle 2**: physics agrees exactly
  until the control output feeds back into it.

### 3. Headless trace (last run before the Python trace format changed)

The golden trace deliberately disables human safety, leaving the safety
geometry, the projection ladder and the router's replanning unverified by it.
This second harness closes that gap: it runs exactly what `srl_sim` runs (configured targets, the configured left-arm trajectory, cylinder routing and
the whole-arm safety filter) for 2,000 closed-loop cycles (4 sim-seconds).

Every **discrete** column matches exactly across all 4,000 samples:

- `reason` (the safety ladder's branch: `clear` / `filtered` /
  `joint_limit_filtered` / the hold reasons),
- `stopped`, `human_adjusted`, `limit_adjusted`, `target_adjusted`,
- `active_count` (how many of the 18 spheres per arm are constraining),
- `route_kind` and `waypoint_count`.

So the two implementations do not merely produce
similar numbers, they take the **same branch** at every decision point, on
every cycle, in a scenario where the safety filter is actively engaging
(`reason=filtered` on the left arm throughout).

`qp_fallback_entries=0`: the OSQP fallback was never reached in 2,000 cycles; the Gauss–Seidel repair sweep resolved every case. OSQP is still built and
wired (`-DSRL_WITH_OSQP=ON`, the default) because the Python would call it if
repair ever failed.

### 4. Trajectory generation

Fixture `cpp/tests/fixtures/trajectory_control.toml` exercises all four
segment types in one program: `hold` -> `line` -> 4-point `waypoints`
(the global C² minimum-jerk spline, which drives the coupled interior-knot
solve) -> 2-revolution clockwise `circle` in the yz plane, with Cartesian
speed/acceleration limits set so the validation path runs.

Bit-identical: `duration`, all segment `boundary` times, the full
`rate_bounds` (including the polynomial-root maxima computed through a
companion-matrix eigensolve), every rotation matrix, every angular velocity
and every angular acceleration.

Residual only in translation: position 3.33e-15, velocity 4.01e-15,
acceleration 1.13e-14, from the quintic coefficient solve.

## Where the residual comes from

1. **MuJoCo and Pinocchio agree exactly.** Same binaries, same call order.
   `torso_position_m` and `sample_time_s` are bit-identical for all 500 rows,
   and on the pre-tuning trace `q` was bit-identical until control feedback
   reached it.
2. **Dense matrix products differ by one ULP.** `R_W_B * J_B` (3x3 times 3x7)
   and `T_W_B * T_B_E` (4x4) go through Eigen's GEMM in C++ and OpenBLAS/
   Accelerate's `cblas_dgemm` in NumPy. Different blocking, same mathematics,
   ~5e-16 apart.
3. **The DLS solve amplifies it.** `qdot_task = Jᵀ (J Jᵀ + λ²I)⁻¹ ẋ` with
   λ = 0.05 has a condition number around 10³ for this arm, turning a 5e-16
   input perturbation into ~2e-13 on the output at the original gains and
   ~2e-12 at the tuned ones. That accounts for the entire
   worst-case figure.

`np.linalg.solve` and `np.linalg.pinv` themselves are *not* a source of
divergence: the port calls Accelerate's `dgesv` and `dgesdd` directly, which
is the same LAPACK NumPy is built on.

### What would close the remaining gap

Routing the handful of control-path matrix products through Accelerate's
`cblas_dgemm` instead of Eigen would very likely make `J` and `ee_pose`
bit-identical and collapse the `qdot_task` residual with them. It was not
done because the residual is a few 1e-12, which is round-off, and the change
would trade readable Eigen expressions for hand-rolled BLAS calls throughout
the kinematics.

## Unit tests

10 suites. The six below were written with the port (336 assertions, each
naming the behaviour-contract clause it defends); `test_cartesian_planner`,
`test_joint_trajectory`, `test_joint_tracking` and `test_planned_runner`
were added later with the planning layer.

| Suite | Checks | Covers |
|---|---|---|
| `test_math` | 20 | A5, D7, D9: rpy composition, quaternions, analytic derivatives, `dgesv`, Moore–Penrose conditions, null-space projector |
| `test_actuation` | 84 | F1–F5: clamp order, exact clamp-free velocity interval, saturation flags, command persistence, infinite bounds |
| `test_safety` | 56 | E1–E12: corner distance rule, on-axis degeneracy, mount exemption, and every rung of the projection ladder including the hold |
| `test_router` | 49 | G1–G8: inflation, route radius, height-band clipping, interior-target adjustment, follower advance, 32-waypoint cap |
| `test_trajectory` | 99 | H1–H10: minimum-jerk endpoints, SO(3) log round-trip incl. the near-pi branch, exact knots, circle geometry, C² program validation, limit scaling |
| `test_config` | 28 | I1–I4: unknown/missing keys, ranges, cross-field rules, disabled-feature validation, CPython float repr, provenance stability |

Two of these caught real mistakes during development: one wrong test
expectation (clearance sign after subtracting sphere radii) and one genuine
conditioning limit near a π rotation, which the Python shares (measured
1.0e-09 there too) and which is now recorded with a justified tolerance
rather than a loosened one.

## Still requiring validation

1. **The OSQP fallback path has never executed.** It is implemented and
   compiled, but 2,000 default-config cycles and 250 golden-trace cycles all
   resolved in the repair sweep. Its numerical agreement with the Python
   `osqp` wrapper is therefore **unverified**. Forcing it (a scenario with a
   genuinely infeasible repair) is the outstanding test.
2. **The `stopped` / hold branches are unit-tested but not observed in a
   closed loop.** No run so far has produced a real safety stop, so the
   hold-and-report behaviour is verified only against constructed inputs.
3. **Long-horizon divergence is unmeasured.** Parity is established over
   0.5 sim-seconds today (4 s in the last headless run). Since the loop is closed, the ~1e-12 residual is fed back;
   whether it stays bounded over minutes has not been characterised.
4. **The viewer is visually unverified.** It runs the same closed loop and
   draws the same overlay geometry, but no image comparison was made against
   the Python `launch_passive` viewer.
5. **`start = "configured_target"`** is implemented but untested; the shipped
   config uses `start = "measured"`.
6. **Legacy `shape = "measured_start_displacement"` trajectories are
   rejected** by the C++ loader rather than translated (see deviation 6).
7. **Single-platform.** Everything here is macOS/Accelerate. On Linux, NumPy
   would use OpenBLAS and the LAPACK-matching argument in §"Where the
   residual comes from" no longer holds; residuals would likely grow.
