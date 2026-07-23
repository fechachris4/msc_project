# Migration execution log

This is the running evidence log for migration Steps 1–6. The execution
contract changed after Step 2: Steps 3–6 run continuously without review
pauses, while retaining per-step tests, golden comparison, independent review,
and a numbered commit.

## Step 1 — freeze current behavior

- Commits: `5900176`, `ba5c9bf`
- Full suite: 161 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS
- Boring choices:
  - Version the trace and manifest directly under `tests/golden/`.
  - Record task velocity, null objective, projected null velocity, final
    controller output, and applied command so a later divergence is local.

## Step 2 — immutable TOML configuration

- Commit: `1d7d277`
- Full suite: 171 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS
- Boring choices:
  - Delete live tuning; edit TOML and restart.
  - Keep offline sweeps immutable and run-local, with effective values stamped.
  - Use `runtime_config.py` beside `config/control.toml` to avoid a Python
    module/package name collision.

## Step 3 — explicit state and world-frame boundary

- Commit: `2a9fb42`
- Full suite: 178 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS after correcting live target integration
- Boring choices:
  - Use fixed, frozen records with read-only NumPy arrays; each record maps
    directly to a C++ struct with fixed Eigen members.
  - Keep all frame selection in `frames.resolve_target_world`; controllers see
    only `WorldTarget`.
  - Read fixed `T_T_B` calibration from MuJoCo model constants, then pass it
    explicitly to pure frame math.
  - Encode the frozen baseline target as its equivalent world pose in TOML;
    retained torso/base targets are re-resolved from each plant sample and
    include transport twist.
- Deviations: none.

## Step 4 — pure controller and reconstruction-only state

- Commit: `97ed090`
- Full suite: 182 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS
- Boring choices:
  - Keep one explicit `ReactivePositionPipeline` for the current
    pose-to-joint-position flow; future direct-velocity or MPC flows are not
    routed through it.
  - Put pose/twist error, PD task twist, DLS IK, and null-space centering in
    `reactive_pose.py`; no second numerical implementation remains in
    diagnostics or the runtime path.
  - Give `PositionIntegrator` sole ownership of the persistent position
    command. It is seeded from measured joints at construction and has no
    reset method.
  - Derive MuJoCo joint/actuator facts in `sim.world`, then pass immutable
    setup records into the controller pipeline.
  - Retain `ControlTrace` field order so existing telemetry and the frozen
    golden schema remain unchanged.
- Deviations: none.

## Step 5 — explicit Runner and MuJoCo backend

- Full suite: 186 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS after routing the experiment and thesis-metric
  paths through the Runner
- Boring choices:
  - Use the three-operation common contract `takeover`, `exchange`, and
    `release`. Initial state arrives from takeover; every later state is the
    reply to exactly one complete command.
  - Let MuJoCo own `data.ctrl`, `mj_step`, scripted torso refresh, and the next
    state read inside `exchange`; the Runner never inspects a backend type.
  - Keep simulation-only model/data inspection public for viewers, contact
    telemetry, and ground-truth tests, but outside the controller contract.
  - Make the current Runner explicitly reactive-pose-to-position instead of
    adding controller-mode flags or a universal pipeline.
  - Represent selected-arm traces as one fixed dual-arm record with optional
    entries rather than a dict-based state contract.
  - Schedule settle-to-evaluation torso motion in an explicit simulation
    scenario object. The same Runner and position integrator continue across
    the phase boundary; no reset branch was added.
- Deviations:
  - The first review found that `analysis.live` and `base_vs_error` still
    stepped MuJoCo directly. Both were moved to Runner/backend exchange before
    the Step 5 commit; the re-review passed.

## Step 6 — provenance-stamped reactive baseline

- Full suite: 188 tests passed
- Golden gate: 500 rows matched at `rtol=1e-12`, `atol=1e-12`
- Independent review: PASS after correcting the evidence-status wording
- Preview validation:
  - Settling gate passed and the run was accepted under the existing policy.
  - Right position-error mean/RMSE/peak:
    `69.0 / 77.4 / 124.5 mm`.
  - Left position-error mean/RMSE/peak:
    `136.7 / 151.7 / 257.8 mm`.
  - Contact and torso-contact warnings remained visible; contact occupancy was
    `20.15%` and the left arm had negative disturbance rejection.
  - Maximum joint-limit penetration was `0.008974 rad`, inside the existing
    `0.02 rad` policy threshold.
- Boring choices:
  - Validate the current default scripted motion without retuning gains or
    changing the controller.
  - Evaluate two complete periods of the current `0.5 Hz` motion after the
    existing settle phase.
  - Use the `10 mm` settling tolerance only to choose the evaluation start;
    retain and report the raw residual and all evaluation errors.
  - Produce one run artifact plus one figure containing world-frame error
    traces and grouped mean/RMSE/peak bars.
  - Stamp the concrete Runner, controller pipeline, backend, exchange
    operation, MuJoCo timing source, effective TOML, and hashes into both
    metadata and manifest.
- Deviations:
  - The first review found that `PROJECT_MAP.md` called the dirty-worktree
    preview a canonical run. The wording was corrected before commit; the
    clean canonical run remains a post-commit gate.
  - The default scenario is not a strong final baseline: left-arm tracking and
    torso contact require diagnosis in a later behavior-changing task. Step 6
    records this result honestly rather than tuning around it.
