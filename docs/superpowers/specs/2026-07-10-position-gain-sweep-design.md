# Position Gain Sweep Design

## Goal

Create a dedicated, reproducible sweep of `KP_POS` and `KD_POS` under one
fixed base-motion scenario. Keep all other controller gains at their current
values and produce separate right- and left-arm heatmaps for four evaluation
metrics.

This is a position-gain sweep, not a position-only controller. The existing
six-row pose solver and rotational feedback remain active.

## Parameter Grid

The sweep contains 90 independent configurations:

- `KP_POS = [2, 12, 22, 32, 42, 52, 62, 72, 80]` in `1/s`.
- `KD_POS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]`, dimensionless.

At program start, read `KP_ROT`, `KD_ROT`, `K_NULL`, and `DAMPING` from
`controller.servo`. Freeze those values for every episode and persist them in
the output metadata and resume fingerprint. Do not tune or zero them.

## Experiment Scenario

Use an explicit scenario independent of later edits to `sim/motion.py`:

```python
linear_amplitude = np.array([0.18, 0.04, 0.05])
rotational_amplitude = np.array([0.0, 0.0, -0.2])
linear_frequency = 0.5
rotational_frequency = 0.5
```

Linear amplitude is in metres, rotational amplitude is RPY in radians, and
both frequencies are in hertz. Both disturbances have a two-second period.

Before starting the disturbance, use the shared experiment runner's existing
static error-threshold-and-dwell settling procedure. Allow at most 20 seconds
to settle. After settling, evaluate exactly 10 seconds, covering five complete
disturbance cycles. Run both arms in every episode.

## Metrics

Compute each metric independently for the right and left arm over evaluation
samples only:

1. Position-error norm RMSE, in millimetres:
   `1000 * sqrt(mean(sum(e_pos**2, axis=1)))`.
2. Maximum position-error norm, in millimetres:
   `1000 * max(norm(e_pos, axis=1))`.
3. Peak measured joint speed, in radians per second:
   `max(abs(qdot_measured))` over all seven joints and evaluation samples.
4. End-effector linear-speed norm RMSE, in metres per second:
   `sqrt(mean(sum(e_v**2, axis=1)))`. The target is static in the world frame,
   so `e_v = -v_ee`; this uses measured/computed world-frame EE velocity, not
   commanded velocity.

Do not merge the two arms into an average or worst-arm value for these plots.
Persist both arms' metrics in each result row.

## Parallel Execution and Resume

Implement this as a new `analysis/position_gain_sweep.py` entry point. Reuse
`analysis.live.run_experiment` and the existing experiment telemetry rather
than duplicating the control loop.

Use multiprocessing with the `spawn` start method so every worker imports an
independent MuJoCo model/data and controller state. Each submitted job contains
one gain pair and returns one result row. Workers never write shared output.
The parent process alone updates the state file atomically after each completed
episode.

Expose `--workers N`; default to `min(4, os.cpu_count() or 1)`. Also expose
`--fresh` to discard saved state and `--plots-only` to regenerate plots and CSV
without running simulations. Re-running normally resumes matching completed
cells.

The resume fingerprint must include the full gain grid, explicit motion
scenario, settling configuration, evaluation duration, fixed controller gains,
and metric schema. Refuse to mix results when the fingerprint changes; require
`--fresh` for a new experiment.

## Outputs

Write to `analysis/output/position_gain_sweep/`:

- `sweep_state.json`: atomically updated resumable state.
- `summary.csv`: one row per gain pair with status, settling information,
  warnings, and all eight arm-specific metrics.
- `metadata.json`: scenario, grid, fixed gains, units, metric definitions,
  timestep, dependency versions, worker count, timestamp, and Git revision.
- Eight heatmaps: four metrics for each arm.

Heatmaps use `KP_POS` on the horizontal axis and `KD_POS` on the vertical axis.
Each cell shows its numeric value. Use a perceptually uniform sequential colour
map, independent colour scaling per metric and arm, and units in titles and
colour-bar labels. Mark cells that did not settle, encountered non-finite data,
or have missing evaluation samples as invalid rather than mapping them to zero.
Retain contact and joint-limit warnings in the CSV/state so questionable cells
remain auditable.

## Verification

Add focused tests for the exact grids, scenario, metric calculations, resume
fingerprint, parent-only state updates, and heatmap matrix placement. Test the
parallel worker path with a small mocked or smoke workload rather than running
all 90 full episodes in the unit suite.

Run one real single-worker smoke subset through MuJoCo to verify process-safe
execution and artifacts, then compare its result with the same configuration
run through the existing shared experiment runner. Verify all eight figures
visually for axes, units, cell placement, annotations, and invalid-cell
treatment. Finally run the complete existing test suite.
