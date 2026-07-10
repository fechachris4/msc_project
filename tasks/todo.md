- [x] Investigate why `sim/scene.xml` fails around line 6.
- [x] Update line 6 to reference the existing Kinova Gen3 model file.
- [x] Verify the XML is well formed and the referenced model assets exist.

## Review

Root cause: line 6 referenced `/Users/christian/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`, which does not exist on this machine. An existing Kinova Gen3 MJCF was found under `/Users/christian/Projects/Code/mujoco-3.9.0/my_project/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`.

Changed `sim/scene.xml` line 6 to use the existing absolute path. Verification passed with `xmllint --noout sim/scene.xml`, `xmllint --noout /Users/christian/Projects/Code/mujoco-3.9.0/my_project/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`, and MuJoCo `compile sim/scene.xml`.

## Current Debug Plan

- [x] Reproduce `main.py` from the project root with `python3`.
- [x] Check whether the scene path should be root-relative or script-relative.
- [x] Make the smallest fix if the path is wrong.
- [ ] Verify MuJoCo loads the scene.
- [x] Record the result here.

## Current Debug Review

Root cause found: [main.py](/Users/christian/Projects/Code/msc_project/main.py:5) loaded `scene.xml` from the process working directory, but the actual file is [sim/scene.xml](/Users/christian/Projects/Code/msc_project/sim/scene.xml:1). Changed `main.py` to resolve `sim/scene.xml` relative to `main.py`.

Verification passed for `python3 -m py_compile main.py`, `xmllint --noout sim/scene.xml`, and an explicit path existence check. Full MuJoCo runtime verification is blocked in this shell because `python3 main.py` fails at `import mujoco` with `ModuleNotFoundError: No module named 'mujoco'`.

## GitHub Publish Plan

- [x] Check whether this directory is already a git repository.
- [x] Confirm whether to initialize a new repository here.
- [x] Confirm GitHub repository name and visibility.
- [x] Stage the intended project files only.
- [x] Commit with a clear message.
- [x] Push to GitHub.
- [x] Record publish result here.

## GitHub Publish Status

Done: repository initialized and published to `https://github.com/fechachris4/msc_project` (remote `origin`). Earlier "not a git repository / invalid gh token" note is obsolete.

## Kinova FK Completion Plan

- [x] Write a minimal failing verification that imports `controller.kinematics` and checks composed right EE pose against MuJoCo's direct site pose.
- [x] Implement tuple-based rigid transform helpers in `controller/kinematics.py`.
- [x] Complete `right_ee_positions()` using `T_W_T * T_T_KR * T_KR_E` without changing unrelated files.
- [x] Verify the composed pose numerically matches the direct MuJoCo pose.
- [x] Record the result here.

## Kinova FK Completion Review

Implemented rigid transform helpers in `controller/kinematics.py`, switched body pose reads to MuJoCo's `data.xpos`/`data.xmat`, and completed `right_ee_positions()` by deriving the current base-relative end-effector pose from MuJoCo FK and recomposing `T_W_T * T_T_KR * T_KR_E`.

Verification:
- Red check: `conda run -n base python -m unittest tests.test_kinematics` failed on the original `SyntaxError` at the unfinished `T_K_E` line.
- Unit check: `conda run -n base python -m unittest tests.test_kinematics` passed.
- MuJoCo check: after `mj_forward`, composed and direct right EE poses matched with `pos_max_abs = 0.0` and `rot_max_abs = 1.1102230246251565e-16`.

## Codebase Review Plan (2026-07-10)

- [x] Establish repository state and identify the last known-good verification path.
- [x] Map runtime entry points, controller flow, simulation inputs, analysis outputs, and tests.
- [x] Audit controller correctness, especially world-frame references, transforms, Jacobians, units, saturation, and joint limits.
- [x] Audit simulation/motion behavior, configuration portability, and failure handling.
- [x] Audit metrics, plots, and tests for reproducibility and thesis-ready evidence.
- [x] Run the available verification commands and distinguish confirmed failures from review risks.
- [x] Prioritize actionable improvements by severity and effort, with exact file and line references.

## Codebase Review

The world-frame transform/FK/Jacobian core is strong and independently validated. The main improvements are at the experiment boundary:

1. Validate `desired_pos` and the production base-motion scenario together. The current 10 s production rollout peaks at 252.5 mm (right) and 198.9 mm (left), with arm contacts in about 41% of steps, while the rejection test uses smaller, FK-generated feasible targets.
2. Replace fixed 2 s settling with an error-threshold-plus-dwell criterion. At 2 s the static errors remain 149.4/212.9 mm and 39.7/42.9 deg; at 10 s they are 1.2/1.6 mm.
3. Complete and test the thesis metric contract: mean, RMSE, and peak; define norm versus per-axis metrics; handle a static base without division by zero.
4. Persist raw logs, metrics, gains, scenario, timestep, evaluation window, dependency versions, and git revision beside each figure.
5. Fix test isolation: full discovery passes 49 tests, but `python -m unittest tests.test_velocity -v` fails all 3 tests because `sim.motion` is imported after a test has rotated the torso.
6. Clamp position-servo setpoints to the intersection of joint and actuator ranges; the actuator ranges currently permit about 0.0096 rad beyond limited-joint bounds.
7. Resolve MJCF paths from module locations instead of the process CWD; importing `sim.world` outside the repository root currently fails.
8. Refresh stale methodology/status documents and dependencies (`matplotlib` is absent from `requirements.txt`; README/velocity report still describe a P-only or unused-velocity path).

Verification evidence:
- `.venv/bin/python -m unittest discover tests`: 49 tests, OK.
- `.venv/bin/python -m unittest tests.test_velocity -v`: 3 tests, 3 failures (import-order dependence reproduced).
- Production scenario, after a genuine 10 s static settle: 252.5/198.9 mm peak error, 20.2%/37.1% rejection, 41.5%/40.9% contact-step occupancy (right/left).
- `analysis.metrics.stats` with a zero base displacement: `ZeroDivisionError` reproduced.
- Importing `sim.world` from `/private/tmp` with the repository on `sys.path`: scene path load failure reproduced.

## Telemetry-Backed Diagnostic Plotting Implementation (2026-07-10)

- [x] Expose immutable controller telemetry without changing actuator commands.
- [ ] Add a reproducible threshold-and-dwell experiment runner and persisted run artifacts.
- [ ] Implement evaluation-only, gain-segment-aware metrics with static-base safety.
- [ ] Add and verify the nine diagnostic figure families.
- [ ] Unify the live dashboard, gain snapshots, and gain segmentation with the shared runner.
- [ ] Migrate the existing analysis entry points and canonical artifact workflow.
- [ ] Repair FK, velocity, and bandwidth validation figures.
- [ ] Update documentation and complete automated and visual verification.

## Telemetry-Backed Diagnostic Plotting Review

Pending implementation and verification.
