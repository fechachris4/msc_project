# C++ port: architecture, dependencies, build, migration plan

## 1. Guiding decisions

**Link the same binaries Python links.** The venv already ships native
MuJoCo 3.10.0 and Pinocchio 4.0.0. Building against those exact libraries
removes the largest class of parity risk (different FK/Jacobian/`log3`
implementations, different physics versions) before a line is written.

**Match NumPy's LAPACK, not just its intent.** NumPy 2.4.4 here is built on
Apple **Accelerate**. `np.linalg.solve` is LAPACK `dgesv`; `np.linalg.pinv`
is `dgesdd` plus a `max(M,N)·eps·σ_max` cutoff. The port calls those same
Accelerate routines with the same arguments, so equations D7 and D9 are
bit-identical rather than merely close. Eigen is used for storage and cheap
dense arithmetic, **not** for `solve`/`pinv` on the control path.

**Explicit ownership, no import-time singletons.** The Python module-level
`backend = MujocoBackend()` is convenient but was flagged in the project's
own risk list. The C++ port constructs everything explicitly in `main`.

**Keep the module seams.** The Python layout is already a clean
simulation/trajectory/control/config/render/telemetry split. The port keeps
those boundaries one-to-one so the two can be read side by side.

## 2. Dependencies

| Dependency | Version | Source | Why |
|---|---|---|---|
| MuJoCo | 3.10.0 | venv wheel (`site-packages/mujoco`) — headers + `libmujoco.3.10.0.dylib` | Same physics build as Python |
| Pinocchio | 4.0.0 | venv `cmeel.prefix` — headers + dylibs | Same FK/Jacobian/`log3` |
| Boost, coal, urdfdom | as shipped | venv `cmeel.prefix` | Pinocchio's own dependencies |
| Eigen | 3.4.0 | FetchContent (pinned) | Required by Pinocchio headers; not installed on this machine |
| Accelerate | system | macOS framework | `dgesv` / `dgesdd`, identical to NumPy |
| toml++ | 3.4.0 | FetchContent (pinned) | TOML parsing; header-only |
| GLFW | 3.4 | FetchContent (pinned) | Viewer window/context |
| OSQP | 1.0.0 | FetchContent (pinned), optional | Safety-projection fallback |

FetchContent is preferred over `brew install` so the build does not mutate
the developer's machine and the versions are pinned in-tree. Every fetched
dependency can be overridden with `-D<NAME>_ROOT` to use a system copy.

## 3. File structure

```
cpp/
  CMakeLists.txt
  cmake/                 dependency discovery for the venv-shipped libraries
  docs/                  01 architecture, 02 contract, 03 design, 04 parity
  src/
    core/       Types.h            Pose, Twist, ArmJointState, PlantState,
                                   FramedTarget, WorldTarget, commands
                Arm.h              side enum + fixed dual-arm container
    config/     RuntimeConfig.*    strict TOML loader, mirrors runtime_config.py
    math/       Transforms.*       SE(3)/SO(3), rpy, quats, sine helpers
                LinAlg.*           Accelerate dgesv/dgesdd wrappers
                SoLog.*            log3/exp3 (delegates to Pinocchio)
    kinematics/ PinModel.*         Pinocchio model ownership
                Frames.*           state → world kinematics, target resolution
                LinkSpheres.h      generated sphere table
    control/    ReactiveController.*   equations 1–6
                SafetyFilter.*         equation 7 + repair + OSQP
                HumanSafety.*          torso-frame geometry
                PositionActuation.*    limits + persistent integration
                CylinderRouter.*       keep-out routing
                Servo.*                pipeline composition
                Runner.*               cycle ordering
    trajectory/ Trajectory.*       hold / waypoint spline / circle / program
                TrajectoryConfig.* structured-intent compiler
    sim/        MujocoBackend.*    model, data, stepping, lifecycle
                Motion.*           scripted torso disturbance
                TargetMotion.*     scripted EE-target disturbance
                Targets.*          mocap marker read/write
                DesiredPos.*       configured targets + marker display
                TargetTrajectory.* trajectory initialisation
    telemetry/  TraceWriter.*      .17g CSV writer
                Console.*          startup + periodic reporting
    render/     Viewer.*           GLFW + mjr passive viewer, overlays
    app/        main.cpp           viewer entry point
                golden_trace.cpp   parity harness (record/check)
  tests/        unit tests per contract clause
  tools/        compare_trace.py   Python↔C++ numeric comparison
```

## 4. Type mapping

| Python | C++ |
|---|---|
| `Pose(position_m, rotation)` | `struct Pose { Eigen::Vector3d position_m; Eigen::Matrix3d rotation; }` |
| `Twist` | `struct Twist { Eigen::Vector3d linear_m_s, angular_rad_s; }` |
| `ArmJointState` | `Eigen::Vector<double,7>` pair |
| `PlantState` | aggregate with `sample_time_s`, `nominal_dt_s`, torso pose/twist, both arms |
| `DualArm*` records | `template <class T> struct DualArm { T right, left; T& for_arm(Side); }` |
| frozen dataclass + read-only arrays | `const`-correct aggregates returned by value |
| `StrEnum TargetFrame` | `enum class TargetFrame { World, Base, Torso }` |
| `TargetSource` Protocol | `class TargetSource` with virtual `sample`/`sample_kinematics` |
| raising `ValueError` | `throw std::invalid_argument` with the same message text |

The Python records validate aggressively in `__post_init__`. The port keeps
validation at *construction boundaries* (config load, backend read, source
sample) rather than on every internal temporary — same guarantees, without
paying the cost 500 times a second on the control path.

## 5. Numerics plan

| Operation | Python | C++ |
|---|---|---|
| `J Jᵀ + λ²I` solve | `np.linalg.solve` | Accelerate `dgesv_`, column-major, same pivoting |
| `pinv(J)` | `np.linalg.pinv` | Accelerate `dgesdd_` + cutoff `max(M,N)·eps·σ_max`, reconstruct `V Σ⁺ Uᵀ` |
| `svd(J)` (diagnostics) | `np.linalg.svd` | `dgesdd_`, values only |
| `log3` | `pin.log3` | `pinocchio::log3` |
| FK / Jacobians | `pin.*` | same Pinocchio calls in the same order |
| `mj_step` etc. | `mujoco.*` | same C API |
| polynomial eval | `np.polynomial.polynomial.polyval` | explicit Horner in ascending order (identical operation order) |
| `polyroots` | companion-matrix eigenvalues | Eigen `EigenSolver` on the same companion matrix |

Row/column-major is the one real trap: NumPy is row-major, LAPACK is
column-major. `J Jᵀ` is symmetric so the transpose is free there, but the
pinv path transposes explicitly and is unit-tested against NumPy.

## 6. Behavioural deviations (intentional, all visible)

1. **No import-time global backend.** Construction is explicit in `main`.
   Same behaviour, better ownership; called out because the Python default
   instance is observable from scripts.
2. **Validation at boundaries, not on every temporary.** See §4.
3. **Analysis/plotting suite is not ported.** `analysis/` and `plotting/`
   are matplotlib experiment tooling, not the simulation. The port instead
   emits the same underlying numbers as CSV so the existing Python figures
   can be produced from C++ output. Stated as scope, not silently dropped.
4. **The viewer is a GLFW/`mjr` passive viewer**, not
   `mujoco.viewer.launch_passive` (which is Python-only). Overlay geometry
   and camera defaults match; interaction affordances differ.
5. **OSQP is optional at build time** (`-DSRL_WITH_OSQP=OFF` drops it). The
   harness reports whether the QP path was ever reached, so the effect of
   omitting it is measured rather than assumed.

## 7. Staged migration plan

Each stage ends with something runnable and a check that can fail.

| Stage | Content | Gate |
|---|---|---|
| 0 | CMake skeleton, dependency discovery, `mj_version`/Pinocchio smoke test | Builds and links against venv libs |
| 1 | `core/Types`, `math/Transforms`, `math/LinAlg` | Unit tests vs NumPy-generated fixtures (A5, D7, D9) |
| 2 | `config/RuntimeConfig` | Loads `control.toml`; effective JSON + sha256 byte-identical to Python (I1–I4) |
| 3 | `kinematics/PinModel`, `Frames`, `LinkSpheres` | FK/Jacobian vs Python at 1e-12 (B1–B6) |
| 4 | `sim/MujocoBackend`, `Motion`, `Targets` | `PlantState` sequence matches Python for N steps (K1, J2) |
| 5 | `control/*` — reactive law, actuation, human safety, router | Per-stage vectors match Python (D, E, F, G) |
| 6 | `control/Servo`, `Runner` | Cycle ordering; full-loop trace (J1) |
| 7 | `trajectory/*` | Spline/circle/program sampling vs Python (H) |
| 8 | `telemetry`, `app/golden_trace` | **250-cycle trace compared column-by-column** |
| 9 | `render/Viewer`, `app/main` | Viewer runs the same closed loop |
| 10 | Parity report | Measured tolerances per column, documented residuals |

The parity gate at stage 8 is the real acceptance test; stages 1–7 exist so
that a divergence is localised to one module instead of hunted across the
whole loop.

## 8. Build and run (target)

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure     # unit tests
./cpp/build/srl_golden_trace --check               # parity vs Python CSV
./cpp/build/srl_sim both                           # viewer
```
