# Parameter Unit Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the units of user-editable Python parameters visible at their definitions without changing values or runtime behaviour.

**Architecture:** Add compact adjacent comments to the existing core tuning surfaces and dedicated analysis configurations. Preserve existing structure and executable tokens; use semantic labels instead of inventing physical units for mixed-Jacobian numerical conventions.

**Tech Stack:** Python 3, NumPy, MuJoCo, `unittest`, standard-library `tokenize`

## Global Constraints

- Use `m`, `rad`, `rad/s`, `Hz`, `s`, `1/s`, and `dimensionless` consistently.
- Preserve every existing numeric value, including `controller.desired_pos.POSES["right"]["rpy"] == [0.0, 45.0, 190.0]` in radians.
- Do not change runtime behaviour, refactor code, rename parameters, or reformat unrelated lines.
- Do not annotate tests, plotting cosmetics, generated outputs, internal calculation literals, or MuJoCo XML values.
- Preserve all pre-existing working-tree changes in `controller/desired_pos.py`, `sim/motion.py`, and unrelated files.

---

### Task 1: Annotate Core Motion and Controller Parameters

**Files:**
- Modify: `sim/motion.py:26-29`
- Modify: `sim/target_motion.py:82-116`
- Modify: `controller/desired_pos.py:18-21`
- Modify: `controller/servo.py:167-177`

**Interfaces:**
- Consumes: Existing RPY convention `rotation_from_rpy([roll, pitch, yaw]) = Rz @ Ry @ Rx` and SI-unit project rules.
- Produces: Adjacent unit documentation for base motion, target motion, desired poses, joint velocity limits, and actuator setpoint lead.

- [ ] **Step 1: Capture the executable-token baseline**

Run:

```bash
.venv/bin/python - <<'PY'
import io
from pathlib import Path
import tokenize

paths = [
    Path("sim/motion.py"),
    Path("sim/target_motion.py"),
    Path("controller/desired_pos.py"),
    Path("controller/servo.py"),
]
for path in paths:
    tokens = [
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    ]
    Path(f"/tmp/{path.as_posix().replace('/', '_')}.tokens").write_text(repr(tokens))
PY
```

Expected: exit status 0 and four `/tmp/*.tokens` baselines created.

- [ ] **Step 2: Add only the core unit comments**

Apply these exact labels without altering values:

```python
LINEAR_AMPLITUDE = np.array([...])       # m, world xyz
ROTATIONAL_AMPLITUDE = np.array([...])   # rad, roll/pitch/yaw
LINEAR_FREQUENCY = ...                   # Hz
ROTATIONAL_FREQUENCY = ...               # Hz
```

In `sim/target_motion.py`, add a short comment immediately above the public pose/twist function signatures mapping `linear_amplitude` to metres, `linear_frequency` to hertz, `rotational_amplitude` to radians as roll/pitch/yaw, and `rotational_frequency` to hertz. In `controller/desired_pos.py`, add one adjacent mapping comment `pos: m; rpy: rad [roll, pitch, yaw]`. In `controller/servo.py`, label `QDOT_LIMIT` as `rad/s` after conversion by `np.radians`, and `CTRL_LEAD` as `rad`.

- [ ] **Step 3: Prove core executable tokens did not change**

Run the same token extraction in memory and compare it with each `/tmp` baseline:

```bash
.venv/bin/python - <<'PY'
import ast
import io
from pathlib import Path
import tokenize

paths = [
    Path("sim/motion.py"),
    Path("sim/target_motion.py"),
    Path("controller/desired_pos.py"),
    Path("controller/servo.py"),
]
for path in paths:
    current = [
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    ]
    baseline = ast.literal_eval(
        Path(f"/tmp/{path.as_posix().replace('/', '_')}.tokens").read_text()
    )
    assert current == baseline, f"executable tokens changed: {path}"
print("core executable tokens unchanged")
PY
```

Expected: `core executable tokens unchanged`.

- [ ] **Step 4: Compile and test the core modules**

Run:

```bash
.venv/bin/python -m py_compile sim/motion.py sim/target_motion.py controller/desired_pos.py controller/servo.py
.venv/bin/python -m unittest tests.test_motion tests.test_target_motion tests.test_reference tests.test_servo -v
```

Expected: compilation exits 0 and all tests in the four named modules pass.

- [ ] **Step 5: Commit the verified core comments**

```bash
git add -p sim/motion.py controller/desired_pos.py
git add sim/target_motion.py controller/servo.py
git diff --cached --check
git diff --cached
git commit -m "docs: label core controller parameter units"
```

At each `git add -p` prompt, stage only the added unit-comment lines. Reject hunks containing the user's pre-existing numeric edits. Confirm in `git diff --cached` that `LINEAR_AMPLITUDE`, `ROTATIONAL_AMPLITUDE`, `LINEAR_FREQUENCY`, `ROTATIONAL_FREQUENCY`, and both `POSES` value lists are absent from changed `-`/`+` lines. Do not absorb those numeric edits into this commit without explicit authorization.

### Task 2: Annotate Experiment and Diagnostic Parameters

**Files:**
- Modify: `analysis/live.py:20-31`
- Modify: `analysis/gain_sweep.py:83-129`
- Modify: `analysis/diagnose.py:28-33`
- Modify: `analysis/base_vs_error.py:47`
- Modify: `analysis/bandwidth_sweep.py:34-48`
- Modify: `analysis/validate_velocity.py:36-45`
- Modify: `plotting/gain_panel.py:24-32`

**Interfaces:**
- Consumes: Gain meanings from `controller/servo.py` and motion units from `sim/motion.py`.
- Produces: Consistent units beside reusable experiment configuration fields, scenario values, sweep grids, settling windows, and live gain controls.

- [ ] **Step 1: Capture analysis executable-token baselines**

Run the Task 1 token-baseline script with these paths:

```python
paths = [
    Path("analysis/live.py"),
    Path("analysis/gain_sweep.py"),
    Path("analysis/diagnose.py"),
    Path("analysis/base_vs_error.py"),
    Path("analysis/bandwidth_sweep.py"),
    Path("analysis/validate_velocity.py"),
    Path("plotting/gain_panel.py"),
]
```

Expected: exit status 0 and seven `/tmp/*.tokens` baselines created.

- [ ] **Step 2: Add only experiment and diagnostic unit comments**

Use these exact semantic mappings:

```text
linear amplitude: m, world xyz
rotational amplitude: rad, roll/pitch/yaw
frequency: Hz
evaluation, settling, dwell, timeout, motion, simulation, and window durations: s
position and rotation proportional gains, plus null-space gain: 1/s
position and rotation derivative gains: dimensionless
scale and guard factors: dimensionless
joint velocity: rad/s
joint margin: rad
DAMPING: DLS lambda
singular-value thresholds: mixed-Jacobian numerical thresholds
```

For arrays created with `np.radians([...])`, label the input literals as degrees converted to radians. For `HOME` joint vectors already expressed as decimal radians, label them `rad`. Preserve comments that already state an equivalent unit and do not duplicate them.

- [ ] **Step 3: Prove analysis executable tokens did not change**

Run the Task 1 comparison script with the seven Task 2 paths.

Expected: `analysis executable tokens unchanged`.

- [ ] **Step 4: Compile and run analysis-related verification**

Run:

```bash
.venv/bin/python -m py_compile analysis/live.py analysis/gain_sweep.py analysis/diagnose.py analysis/base_vs_error.py analysis/bandwidth_sweep.py analysis/validate_velocity.py plotting/gain_panel.py
.venv/bin/python -m unittest discover tests -v
```

Expected: compilation exits 0 and the complete existing test suite passes.

- [ ] **Step 5: Review scope and commit the analysis comments**

Run:

```bash
git diff --check
git diff --stat
git diff -- analysis plotting/gain_panel.py
```

Expected: only adjacent comments/docstrings changed in the listed files; no values, names, or executable statements changed.

Then commit:

```bash
git add analysis/live.py analysis/gain_sweep.py analysis/diagnose.py analysis/base_vs_error.py analysis/bandwidth_sweep.py analysis/validate_velocity.py plotting/gain_panel.py
git diff --cached --check
git commit -m "docs: label analysis parameter units"
```

### Task 3: Final Verification and Task Record

**Files:**
- Modify: `tasks/todo.md`

**Interfaces:**
- Consumes: Verified Task 1 and Task 2 comment-only commits.
- Produces: Durable verification record for the repository workflow.

- [ ] **Step 1: Verify the repository-wide unit-comment scope**

Run:

```bash
rg -n "^(LINEAR_AMPLITUDE|ROTATIONAL_AMPLITUDE|LINEAR_FREQUENCY|ROTATIONAL_FREQUENCY|QDOT_LIMIT|CTRL_LEAD|SETTLE_SECONDS|SIM_SECONDS|MOTION_SECONDS|KP_POS_GRID|KD_POS_GRID|KP_ROT_GRID|KD_ROT_GRID|DAMPING_GRID|K_NULL_GRID)\s*=" sim controller analysis
git diff --check HEAD~2..HEAD
```

Expected: each listed editable parameter definition has an adjacent unit or semantic comment, and the two implementation commits have no whitespace errors.

- [ ] **Step 2: Run final verification**

Run:

```bash
.venv/bin/python -m unittest discover tests
```

Expected: the full suite passes with no failures or errors.

- [ ] **Step 3: Record the result**

In `tasks/todo.md`, check the three remaining implementation items and replace `Design approved; implementation pending written-spec review.` with a concise review listing:

```text
Added adjacent unit comments to the core tuning and analysis configuration surfaces. No parameter values or executable tokens changed. Verification: Python compilation passed for all touched modules; the full unittest discovery suite passed; git diff --check passed.
```

Use the actual test count and any justified deviations from the planned commands.

- [ ] **Step 4: Commit the task record**

```bash
git add tasks/todo.md
git diff --cached --check
git commit -m "docs: record parameter unit verification"
```
