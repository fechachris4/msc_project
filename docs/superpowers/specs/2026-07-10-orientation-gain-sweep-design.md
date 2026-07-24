# Orientation Gain Sweep Design

## Goal

Create a dedicated, reproducible sweep of `KP_ROT` and `KD_ROT` under the
same mixed base-motion scenario as the position-gain sweep. Keep all other
controller gains at their current values and produce separate right- and
left-arm heatmaps for four orientation-focused evaluation metrics.

This is an orientation-gain sweep, not an orientation-only controller. The
existing six-row world-frame pose solver and positional feedback remain active.

## Parameter Grid

The sweep contains 90 independent configurations:

- `KP_ROT = [2, 12, 22, 32, 42, 52, 62, 72, 80]` in `1/s`.
- `KD_ROT = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]`, dimensionless.

At program start, read `KP_POS`, `KD_POS`, `K_NULL`, and `DAMPING` from
`controller.servo`. Freeze those values for every episode and persist them in
the output metadata and resume fingerprint. Do not tune or zero them.

## Experiment Scenario

Use the exact explicit mixed scenario from the position-gain sweep, independent
of later edits to `sim/motion.py`:

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
to settle. After settling, evaluate exactly 4 seconds, covering two complete
disturbance cycles. Run both arms in every episode.

## Metrics

Compute each metric independently for the right and left arm over evaluation
samples only:

1. Orientation-error norm RMSE, in degrees:
   `degrees(sqrt(mean(sum(e_rot**2, axis=1))))`.
2. Maximum orientation-error norm, in degrees:
   `degrees(max(norm(e_rot, axis=1)))`.
3. Peak measured joint speed, in radians per second:
   `max(abs(qdot_measured))` over all seven joints and evaluation samples.
4. End-effector angular-speed error norm RMSE, in radians per second:
   `sqrt(mean(sum(e_w**2, axis=1)))`. The target is static in the world frame,
   so `e_w = -w_ee`; this uses measured/computed world-frame EE angular
   velocity, not commanded velocity.

Do not merge the two arms into an average or worst-arm value for these plots.
Persist both arms' metrics in each result row.

## Architecture, Parallel Execution, and Resume

Implement a new `analysis/orientation_gain_sweep.py` entry point by mirroring
the proven position-sweep structure. Do not refactor or alter the existing
position sweep. Reuse `analysis.live.run_experiment` and existing telemetry
rather than duplicating the control loop.

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

Write to `analysis/output/orientation_gain_sweep/`:

- `sweep_state.json`: atomically updated resumable state.
- `summary.csv`: one row per gain pair with status, settling information,
  warnings, and all eight arm-specific metrics.
- `metadata.json`: scenario, grid, fixed gains, units, metric definitions,
  timestep, dependency versions, worker count, timestamp, and Git revision.
- Eight heatmaps: four metrics for each arm.

Heatmaps use `KP_ROT` on the horizontal axis and `KD_ROT` on the vertical axis.
Each cell shows its numeric value. Use a perceptually uniform sequential colour
map, independent colour scaling per metric and arm, and units in titles and
colour-bar labels. Mark cells that did not settle, encountered non-finite data,
or have missing evaluation samples as invalid rather than mapping them to zero.
Retain contact and joint-limit warnings in the CSV/state so questionable cells
remain auditable.

## Verification

Add focused tests for the exact grids, unchanged mixed scenario, fixed gains,
metric calculations and degree conversion, resume fingerprint, parent-only
state updates, and heatmap matrix placement. Test the parallel worker path with
a small mocked or smoke workload rather than running all 90 full episodes in
the unit suite.

Run one real single-worker smoke subset through MuJoCo to verify process-safe
execution and artifacts, then compare its result with the same configuration
run through the shared experiment runner. Verify all eight figures visually for
axes, units, cell placement, annotations, and invalid-cell treatment. Finally
run the focused relevant tests and the complete existing test suite, recording
any failures that reproduce outside the new files.
