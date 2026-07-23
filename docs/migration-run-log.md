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
