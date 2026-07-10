"""Staged sequential gain sweep for the reactive controller's six servo
gains (controller/servo.py: KP_POS, KP_ROT, KD_POS, KD_ROT, K_NULL,
DAMPING). A full 6-D combination sweep is intractable, so this sweeps
one gain group per stage and carries the winner forward:

    stage 0: baseline gains (1 episode, progression-plot anchor)
    stage 1: KP_POS x KD_POS               (56 episodes)
    stage 2: KP_ROT x KD_ROT                (56 episodes)
    stage 3: uniform scale on the 4 stage-1/2 winner task gains (<=7)
    stage 4: DAMPING x K_NULL               (42 episodes)
    stage 5: reporting only (sweep_progression.png; no new episodes)

Winner = lowest worst-arm position RMSE (rotation RMSE for stage 2),
disqualifying configs that fail to settle, contact the world/torso,
produce non-finite data, penetrate a joint's soft limit past
JOINT_MARGIN_TOL_RAD, saturate a joint over --sat-threshold percent, or
(stages 3-4) regress worst-arm rotation RMSE past ROT_GUARD_FACTOR x the
stage-2 winner. (live.py's own blanket "valid" flag is not used here --
see disqualify_reasons -- since MuJoCo's soft joint limits are meant to
be pushed into a little, and that alone shouldn't fail a config.)

Builds on the headless harness in analysis/live.py (run_experiment,
save_run) and analysis/metrics.py (experiment_metrics) -- no controller
changes. Each episode discards its full per-step log immediately after
extracting metrics (dual-arm 14k-step logs are tens of MB); only the
final winner of each stage is re-run once and persisted via
live.save_run.

    python -m analysis.gain_sweep                  # all stages, auto-resume
    python -m analysis.gain_sweep --stage 2         # one stage (needs stage 1's winner)
    python -m analysis.gain_sweep --fresh           # discard sweep_state.json, start over
    python -m analysis.gain_sweep --smoke           # tiny grids, minutes -- pipeline smoke test
    python -m analysis.gain_sweep --plots-only      # regenerate plots/summary.csv from state
    python -m analysis.gain_sweep --sat-threshold 30

Progress and state are persisted to analysis/output/gain_sweep/
sweep_state.json after every episode (atomic temp-file + os.replace), so
an interrupted run resumes with at most one lost episode. Resuming
refuses (asks for --fresh) if the scenario or grids have changed since
the state was written -- episodes are meaningless outside the
scenario/grid they were run under. A --sat-threshold change alone is
allowed and free: episode results don't depend on it, so resuming
reselects every stage's winner from the stored rows instead of
rerunning anything -- except a stage whose base gains were inherited
from an earlier stage's winner that moved under the new threshold,
which is dropped and reruns from scratch (its later stages cascade the
same way).

Outputs, all under analysis/output/gain_sweep/ (analysis/output/
gain_sweep_smoke/ for --smoke):
    sweep_state.json                     resumable per-episode state
    summary.csv                          every episode, flattened
    stage{1,2,4}_{pos,rot}_heatmap.png   selection metric over the grid
    stage{1,2,4}_metrics_panel.png       mean/RMSE/peak/saturation/settle/headroom
    stage3_scale_curves.png              metrics vs. scale factor, per arm
    sweep_progression.png                baseline -> each stage's winner
    stage{N}_winner_run/...              full save_run() artifacts for each winner
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import live, metrics
from controller import servo
from plotting.style import C_XYZ, SIDE_COLOR, SIDE_STYLE

OUT = Path("analysis/output/gain_sweep")

# The shared base-motion scenario for every episode: 0.3 m x-sinusoid at
# 0.1 Hz (sim/motion.py's own research-lever defaults), both arms,
# evaluation_seconds = 25.0 s = 2.5 periods. settle_timeout stays at
# ExperimentConfig's own default (15.0 s).
SCENARIO = live.ExperimentConfig(
    arms=("right", "left"),
    linear_amplitude=np.array([0.3, 0.0, 0.0]),
    linear_frequency=0.1,
    rotational_amplitude=np.zeros(3),
    rotational_frequency=0.0,
    evaluation_seconds=25.0,
)

SAT_THRESHOLD_PCT = 20.0  # disqualify worst-arm velocity saturation above this
ROT_GUARD_FACTOR = 2.0   # stages 3-4: rot RMSE must stay <= this x stage-2 winner
KD_CEILING = 0.95        # servo.py documents KD < 1 as a stability limit
# MuJoCo soft-limit penetration of ~0.007 rad observed at baseline gains in
# the real scenario (joint 6 rides its limit on both arms, settled and
# otherwise tracking fine) -- constraint compliance, not instability. Only
# deeper violations than this indicate genuine limit crashing.
JOINT_MARGIN_TOL_RAD = -0.02

GAIN_NAMES = ("KP_POS", "KP_ROT", "KD_POS", "KD_ROT", "K_NULL", "DAMPING")
BASELINE_GAINS = {name: float(getattr(servo, name)) for name in GAIN_NAMES}


def _round4(value):
    """Round to 4 significant figures -- stable, short config_ids."""
    return float(f"{float(value):.4g}")


def _fmt_id(value):
    return f"{value:g}"


# --- grids (ranges from plotting/gain_panel.py's sliders) -------------------

KP_POS_GRID = [_round4(v) for v in np.geomspace(0.5, 10.0, 8)]
KD_POS_GRID = [_round4(v) for v in np.linspace(0.0, 0.9, 7)]
KP_ROT_GRID = [_round4(v) for v in np.geomspace(0.5, 10.0, 8)]
KD_ROT_GRID = [_round4(v) for v in np.linspace(0.0, 0.9, 7)]
SCALE_GRID = [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5]
DAMPING_GRID = [_round4(v) for v in np.geomspace(0.002, 0.2, 7)]
K_NULL_GRID = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]


def _use_smoke_mode():
    """Swap in tiny 2x2-ish grids, a short scenario, and a separate
    output directory -- called once from main() before anything reads
    SCENARIO/OUT/the grids, so every downstream function (which reads
    these as plain module globals, same pattern as bandwidth_sweep.py's
    FREQUENCIES) sees the smoke configuration transparently."""
    global SCENARIO, OUT, KP_POS_GRID, KD_POS_GRID, KP_ROT_GRID, KD_ROT_GRID
    global SCALE_GRID, DAMPING_GRID, K_NULL_GRID
    SCENARIO = SCENARIO._replace(linear_frequency=0.5, evaluation_seconds=4.0)
    OUT = Path("analysis/output/gain_sweep_smoke")
    KP_POS_GRID = [_round4(v) for v in np.geomspace(0.5, 10.0, 2)]
    KD_POS_GRID = [_round4(v) for v in np.linspace(0.0, 0.9, 2)]
    KP_ROT_GRID = [_round4(v) for v in np.geomspace(0.5, 10.0, 2)]
    KD_ROT_GRID = [_round4(v) for v in np.linspace(0.0, 0.9, 2)]
    SCALE_GRID = [0.75, 1.3]
    DAMPING_GRID = [_round4(v) for v in np.geomspace(0.002, 0.2, 2)]
    K_NULL_GRID = [0.0, 2.0]


# --- stage grids -------------------------------------------------------


def stage_grid(stage, winners):
    """[(config_id, six-gain dict)] for one stage. winners: {stage_number:
    that stage's winning full six-gain dict}, for stages already
    completed -- stage 1 needs none (starts from BASELINE_GAINS), stage
    2 needs winners[1], stage 3 needs winners[2], stage 4 needs
    winners[3]. config_ids are deterministic (e.g. "s1_kp2.828_kd0.45"),
    used as the resume key in sweep_state.json."""
    if stage == 0:
        return [("s0_baseline", dict(BASELINE_GAINS))]
    if stage == 1:
        base = dict(BASELINE_GAINS)
        return [
            (f"s1_kp{_fmt_id(kp)}_kd{_fmt_id(kd)}",
             dict(base, KP_POS=kp, KD_POS=kd))
            for kp in KP_POS_GRID for kd in KD_POS_GRID
        ]
    if stage == 2:
        base = dict(winners[1])
        return [
            (f"s2_kp{_fmt_id(kp)}_kd{_fmt_id(kd)}",
             dict(base, KP_ROT=kp, KD_ROT=kd))
            for kp in KP_ROT_GRID for kd in KD_ROT_GRID
        ]
    if stage == 3:
        base = dict(winners[2])
        grid = []
        for scale in SCALE_GRID:
            kd_pos = _round4(base["KD_POS"] * scale)
            kd_rot = _round4(base["KD_ROT"] * scale)
            if kd_pos >= KD_CEILING or kd_rot >= KD_CEILING:
                continue
            gains = dict(
                base,
                KP_POS=_round4(base["KP_POS"] * scale), KD_POS=kd_pos,
                KP_ROT=_round4(base["KP_ROT"] * scale), KD_ROT=kd_rot,
            )
            grid.append((f"s3_scale{_fmt_id(scale)}", gains))
        return grid
    if stage == 4:
        base = dict(winners[3])
        return [
            (f"s4_damp{_fmt_id(damping)}_knull{_fmt_id(knull)}",
             dict(base, DAMPING=damping, K_NULL=knull))
            for damping in DAMPING_GRID for knull in K_NULL_GRID
        ]
    raise ValueError(f"unknown stage {stage!r}")


def _parse_stage3_scale(config_id):
    return float(config_id.split("scale", 1)[1])


# --- one episode ---------------------------------------------------------


def _headroom_series(qdot_raw_eval):
    """Per-step max_i |qdot_raw_i| / QDOT_LIMIT_i over a (steps, 7) batch
    -- analysis.dashboard.headroom_frac, vectorized (tests/
    test_gain_sweep.py checks the two agree per step). Discriminates
    configs that all show 0% clipped saturation but differ in how close
    to the limit they run."""
    return np.max(np.abs(qdot_raw_eval) / servo.QDOT_LIMIT, axis=-1)


def _worst_over_arms(arms_metrics, key):
    values = [side[key] for side in arms_metrics.values()
              if side[key] is not None]
    return max(values) if values else None


def _min_over_arms(arms_metrics, key):
    """Like _worst_over_arms, but "worst" is the most negative value --
    used for joint_margin_min_rad, where negative means limit
    penetration (unlike the error/saturation metrics, where worst is
    the largest)."""
    values = [side[key] for side in arms_metrics.values()
              if side[key] is not None]
    return min(values) if values else None


def _finite_or_none(value):
    """metrics.py's _finite_float convention, duplicated locally: a
    non-finite computed value (e.g. headroom from an unstable corner's
    NaN qdot_raw) becomes JSON-safe None rather than NaN. Episodes with
    non-finite data are already flagged invalid by live.py's own
    "non-finite data" warning and so get disqualified anyway -- this
    only prevents write_state_atomic's allow_nan=False json.dumps from
    crashing (and re-crashing on every resume) on the way there."""
    value = float(value)
    return value if np.isfinite(value) else None


def _episode_row(log, gains):
    mask = log.evaluation_mask
    computed = metrics.experiment_metrics(log)
    arms_metrics = {
        side: {
            "position_error_norm_mean_m":
                computed["arms"][side]["position_error_norm_mean_m"],
            "position_error_norm_rmse_m":
                computed["arms"][side]["position_error_norm_rmse_m"],
            "position_error_norm_peak_m":
                computed["arms"][side]["position_error_norm_peak_m"],
            "rotation_error_norm_mean_rad":
                computed["arms"][side]["rotation_error_norm_mean_rad"],
            "rotation_error_norm_rmse_rad":
                computed["arms"][side]["rotation_error_norm_rmse_rad"],
            "rotation_error_norm_peak_rad":
                computed["arms"][side]["rotation_error_norm_peak_rad"],
            "velocity_saturation_overall_pct":
                computed["arms"][side]["velocity_saturation_overall_pct"],
            "joint_margin_min_rad":
                computed["arms"][side]["joint_margin_min_rad"],
        }
        for side in log.arms
    }
    headroom_means = []
    headroom_p95s = []
    for side in log.arms:
        qdot_raw = np.asarray(log.arm_data[side]["qdot_raw"], dtype=float)[mask]
        if qdot_raw.size == 0:
            continue
        series = _headroom_series(qdot_raw)
        mean_val = _finite_or_none(np.mean(series))
        p95_val = _finite_or_none(np.percentile(series, 95.0))
        if mean_val is not None:
            headroom_means.append(mean_val)
        if p95_val is not None:
            headroom_p95s.append(p95_val)
    headroom_mean = max(headroom_means) if headroom_means else None
    headroom_p95 = max(headroom_p95s) if headroom_p95s else None
    return {
        "gains": dict(gains),
        "settled": bool(log.settled),
        "settle_duration_s": _finite_or_none(log.settle_duration),
        "valid": bool(log.valid),
        "warning_reasons": list(log.warning_reasons),
        "arms": arms_metrics,
        "velocity_saturation_overall_pct": _worst_over_arms(
            arms_metrics, "velocity_saturation_overall_pct"),
        "headroom_mean_frac": headroom_mean,
        "headroom_p95_frac": headroom_p95,
        "worst_arm_pos_rmse_m": _worst_over_arms(
            arms_metrics, "position_error_norm_rmse_m"),
        "worst_arm_rot_rmse_rad": _worst_over_arms(
            arms_metrics, "rotation_error_norm_rmse_rad"),
        "worst_arm_joint_margin_min_rad": _min_over_arms(
            arms_metrics, "joint_margin_min_rad"),
    }


def run_episode(gains):
    """Run one gain configuration through the shared SCENARIO and return
    a flat selection-metrics row; the per-step log itself is discarded."""
    log = live.run_experiment(SCENARIO, gains=gains)
    return _episode_row(log, gains)


# --- disqualification and winner selection --------------------------------


# live.py's ExperimentLog.valid is a blanket flag: any negative joint
# margin at all makes it False, but MuJoCo's soft joint limits are meant
# to be pushed into a little (see JOINT_MARGIN_TOL_RAD above) -- so
# disqualification checks specific warning_reasons instead of the
# blanket flag. "settling timeout" is covered by the separate "not
# settled" check; "negative joint margin" is superseded by the
# JOINT_MARGIN_TOL_RAD check below (which only fires on genuinely deep
# penetration); "evaluation shorter than one disturbance period" is a
# scenario-shape warning, irrelevant to gain selection.
_DISQUALIFYING_WARNINGS = (
    "contact detected", "torso contact detected", "non-finite data")


def disqualify_reasons(row, stage, sat_threshold=SAT_THRESHOLD_PCT,
                        stage2_winner_rot_rmse=None):
    """Reasons row's config must not win its stage; empty = qualified.
    stage2_winner_rot_rmse: stage 2's winning worst-arm rotation RMSE
    (rad) -- required only for the stage 3/4 rotation-regression guard,
    ignored otherwise."""
    reasons = []
    if not row["settled"]:
        reasons.append("not settled")
    for warning in _DISQUALIFYING_WARNINGS:
        if warning in row["warning_reasons"]:
            reasons.append(warning)
    margin = row.get("worst_arm_joint_margin_min_rad")
    if margin is not None and margin < JOINT_MARGIN_TOL_RAD:
        reasons.append(
            f"joint limit penetration {np.degrees(margin):.2f} deg exceeds "
            f"tolerance {np.degrees(JOINT_MARGIN_TOL_RAD):.2f} deg")
    sat = row["velocity_saturation_overall_pct"]
    if sat is not None and sat > sat_threshold:
        reasons.append(
            f"worst-arm saturation {sat:.1f}% > {sat_threshold:.1f}% threshold")
    if stage in (3, 4) and stage2_winner_rot_rmse is not None:
        rot = row["worst_arm_rot_rmse_rad"]
        guard = ROT_GUARD_FACTOR * stage2_winner_rot_rmse
        if rot is not None and rot > guard:
            reasons.append(
                f"rotation RMSE {rot:.4f} rad > {guard:.4f} rad "
                f"({ROT_GUARD_FACTOR:g}x stage-2 winner) guard")
    return reasons


def select_winner(rows, metric_key, strict=True):
    """Lowest metric_key among qualified (non-disqualified) rows. Tie
    chain: metric -> worst-arm saturation -> settle time -> grid order
    (rows' own order). strict=False returns None instead of raising when
    every row is disqualified -- used only to star-annotate plots, which
    must not blow up a stage whose authoritative selection later fails.
    strict=True (the default, used for the real winner pick) prints the
    5 best rows with their reasons and raises RuntimeError: never
    silently promote a disqualified config, since later stages inherit
    it."""
    qualified = [(i, row) for i, row in enumerate(rows)
                 if not row["disqualification"]]
    if not qualified:
        if not strict:
            return None
        ranked = sorted(
            rows,
            key=lambda row: (row[metric_key] is None,
                              row[metric_key] if row[metric_key] is not None
                              else float("inf")))
        stage = rows[0]["stage"] if rows else "?"
        print(f"stage {stage}: all {len(rows)} configs disqualified. "
              "5 best by metric (regardless of disqualification):")
        for row in ranked[:5]:
            print(f"  {row['config_id']}: {metric_key}={row[metric_key]!r}  "
                  f"reasons={row['disqualification']}")
        raise RuntimeError(
            f"stage {stage}: every config was disqualified; relax "
            "--sat-threshold or inspect the reasons printed above, then "
            "resume (already-run episodes are kept in sweep_state.json)")

    def sort_key(item):
        i, row = item
        metric = row[metric_key] if row[metric_key] is not None else float("inf")
        sat = (row["velocity_saturation_overall_pct"]
               if row["velocity_saturation_overall_pct"] is not None
               else float("inf"))
        settle = (row["settle_duration_s"]
                  if row["settle_duration_s"] is not None else float("inf"))
        return (metric, sat, settle, i)

    qualified.sort(key=sort_key)
    return qualified[0][1]


# --- resumable state -------------------------------------------------------


def _grid_fingerprint():
    payload = json.dumps({
        "kp_pos": KP_POS_GRID, "kd_pos": KD_POS_GRID,
        "kp_rot": KP_ROT_GRID, "kd_rot": KD_ROT_GRID,
        "scale": SCALE_GRID, "kd_ceiling": KD_CEILING,
        "damping": DAMPING_GRID, "k_null": K_NULL_GRID,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _stage_grid_signature(grid):
    """Hash of a stage_grid() result's full (config_id, gains) pairs --
    not just the config_ids. Stage 2's config_ids only encode KP_ROT/
    KD_ROT (not the KP_POS/KD_POS it inherits from stage 1's winner), so
    two different stage-1 winners can produce identical config_ids with
    different gains; hashing the gains too is what lets
    _reconcile_state_for_threshold_change detect that a dependent
    stage's base actually changed."""
    payload = json.dumps([[cid, gains] for cid, gains in grid], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _new_state(args):
    return {
        "scenario": live._config_json(SCENARIO),
        "sat_threshold": float(args.sat_threshold),
        "fingerprint": _grid_fingerprint(),
        "stages": {},
    }


def _reconcile_state_for_threshold_change(state, new_threshold):
    """Episode results (settled/valid/RMSE/saturation/...) don't depend
    on sat_threshold -- it's a selection-time filter -- so a threshold
    change alone never requires rerunning episodes: recompute every
    stored row's disqualification and reselect each stage's winner from
    the stored rows, in stage order (1 -> 4) so each stage sees its
    predecessor's already-reconciled winner.

    A stage whose base gains are inherited from an earlier stage's
    winner (2 from 1, 3 from 2, 4 from 3) is invalidated -- its rows and
    winner dropped entirely, forcing a full rerun next time -- if that
    earlier winner changed enough to change this stage's own grid
    (_stage_grid_signature mismatch). Later stages then cascade-drop too,
    since _collected_winners no longer finds a winner to inherit from."""
    total_rows = sum(len(s["rows"]) for s in state["stages"].values())
    print(f"sat threshold {state['sat_threshold']:.1f} -> {new_threshold:.1f}: "
          f"reselecting winners from {total_rows} stored episodes")
    state["sat_threshold"] = new_threshold

    stage2_rot_rmse = None
    for stage in (1, 2, 3, 4):
        stage_key = str(stage)
        stage_state = state["stages"].get(stage_key)
        if not stage_state or not stage_state["rows"]:
            continue

        winners = _collected_winners(state, stage)
        if stage >= 2 and (stage - 1) not in winners:
            print(f"stage {stage}: stage {stage - 1} has no winner under "
                  "the new threshold -- dropping its stored episodes")
            del state["stages"][stage_key]
            continue

        grid = stage_grid(stage, winners)
        if stage >= 2:
            signature = _stage_grid_signature(grid)
            if stage_state.get("grid_signature") != signature:
                print(f"stage {stage}: base gains changed (stage "
                      f"{stage - 1}'s winner moved) -- dropping its stored "
                      "episodes; it will rerun from scratch")
                del state["stages"][stage_key]
                continue

        rows = [stage_state["rows"][cid] for cid, _ in grid
                if cid in stage_state["rows"]]
        for row in rows:
            row["disqualification"] = disqualify_reasons(
                row, stage, sat_threshold=new_threshold,
                stage2_winner_rot_rmse=(
                    stage2_rot_rmse if stage in (3, 4) else None))

        metric_key = "worst_arm_rot_rmse_rad" if stage == 2 else "worst_arm_pos_rmse_m"
        winner = select_winner(rows, metric_key, strict=False) if rows else None
        stage_state["winner_config_id"] = winner["config_id"] if winner else None

        if stage == 2:
            stage2_rot_rmse = (winner["worst_arm_rot_rmse_rad"]
                                if winner is not None else None)


def load_or_init_state(args):
    """Load OUT/sweep_state.json, or start fresh if absent or --fresh.
    Refuses to resume (raises) if the scenario or grids no longer match
    what's stored -- episodes are meaningless outside the scenario/grid
    they were run under, so silently mixing them would be wrong; pass
    --fresh to discard the old state and start over. A --sat-threshold
    change alone is allowed: episode results don't depend on it, so
    _reconcile_state_for_threshold_change reselects winners from the
    stored rows instead of refusing."""
    path = OUT / "sweep_state.json"
    if args.fresh and path.exists():
        path.unlink()
    if not path.exists():
        return _new_state(args)
    stored = json.loads(path.read_text())
    fresh = _new_state(args)
    if (stored.get("fingerprint") != fresh["fingerprint"]
            or stored.get("scenario") != fresh["scenario"]):
        raise RuntimeError(
            f"{path} does not match the current scenario/grids; "
            "pass --fresh to discard it and start over")
    if stored.get("sat_threshold") != fresh["sat_threshold"]:
        _reconcile_state_for_threshold_change(stored, fresh["sat_threshold"])
    return stored


def write_state_atomic(state):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_state.json"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, allow_nan=False))
    os.replace(tmp_path, path)


def _collected_winners(state, stage):
    winners = {}
    for s in range(1, stage):
        stage_state = state["stages"].get(str(s))
        if stage_state and stage_state.get("winner_config_id"):
            winners[s] = stage_state["rows"][stage_state["winner_config_id"]]["gains"]
    return winners


# --- summary.csv -----------------------------------------------------------


def _write_summary_csv(state):
    OUT.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for stage_key in sorted(state["stages"], key=int):
        for row in state["stages"][stage_key]["rows"].values():
            flat_rows.append(metrics.flatten_metrics(row))
    if not flat_rows:
        return
    fieldnames = sorted({key for flat in flat_rows for key in flat})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


# --- plots -------------------------------------------------------------


def _edges(values, log):
    """Cell edges for pcolormesh from a 1-D array of (possibly irregular,
    e.g. geomspace) cell centers -- geometric mean between neighbors for
    a log axis, arithmetic mean for a linear one, with the outer edges
    extrapolated the same way."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        v = values[0]
        return np.array([v * 0.9, v * 1.1]) if log \
            else np.array([v - 0.5, v + 0.5])
    if log:
        mids = np.sqrt(values[:-1] * values[1:])
        first = values[0]**2 / mids[0]
        last = values[-1]**2 / mids[-1]
    else:
        mids = 0.5 * (values[:-1] + values[1:])
        first = 2.0 * values[0] - mids[0]
        last = 2.0 * values[-1] - mids[-1]
    return np.concatenate([[first], mids, [last]])


def _grid_matrices(rows, n_outer, n_inner, metric_key, unit_scale):
    """rows must be in stage_grid's nested (outer, inner) enumeration
    order. Returns (values, disqualified), both shape (n_inner, n_outer)
    to match pcolormesh(x=outer, y=inner)."""
    values = np.full((n_inner, n_outer), np.nan)
    disqualified = np.zeros((n_inner, n_outer), dtype=bool)
    for k, row in enumerate(rows):
        i, j = divmod(k, n_inner)
        metric = row[metric_key]
        if metric is not None:
            values[j, i] = metric * unit_scale
        disqualified[j, i] = bool(row["disqualification"])
    return values, disqualified


def _heatmap(x_grid, y_grid, x_label, y_label, values, disqualified,
             winner_xy, cbar_label, title, log_x, path):
    x_edges = _edges(x_grid, log_x)
    y_edges = _edges(y_grid, False)
    masked = np.ma.masked_array(values, mask=disqualified | np.isnan(values))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")

    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, shading="flat")
    for j, y in enumerate(y_grid):
        for i, x in enumerate(x_grid):
            if np.isnan(values[j, i]):
                continue
            ax.text(x, y, f"{values[j, i]:.1f}", ha="center", va="center",
                    fontsize=6, color="white")
    if log_x:
        ax.set_xscale("log")
    if winner_xy is not None:
        ax.plot(winner_xy[0], winner_xy[1], marker="*", markersize=18,
                color="gold", markeredgecolor="black", linestyle="none",
                label="winner")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _metrics_panel(stage, rows, x_grid, y_grid, x_key, y_key, log_x):
    is_rot = (stage == 2)
    prefix = "rotation_error_norm" if is_rot else "position_error_norm"
    suffix = "rad" if is_rot else "m"
    unit_scale = float(np.degrees(1.0)) if is_rot else 1000.0
    unit = "deg" if is_rot else "mm"

    panels = [
        (f"{prefix}_mean_{suffix}", f"mean [{unit}]", unit_scale, True),
        (f"{prefix}_rmse_{suffix}", f"RMSE [{unit}]", unit_scale, True),
        (f"{prefix}_peak_{suffix}", f"peak [{unit}]", unit_scale, True),
        ("velocity_saturation_overall_pct", "saturation [%]", 1.0, False),
        ("settle_duration_s", "settle time [s]", 1.0, False),
        ("headroom_mean_frac", "headroom, mean [frac]", 1.0, False),
    ]
    n_outer, n_inner = len(x_grid), len(y_grid)
    x_edges = _edges(x_grid, log_x)
    y_edges = _edges(y_grid, False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (key, label, scale, per_arm) in zip(axes.flat, panels):
        values = np.full((n_inner, n_outer), np.nan)
        for k, row in enumerate(rows):
            i, j = divmod(k, n_inner)
            metric = _worst_over_arms(row["arms"], key) if per_arm else row[key]
            if metric is not None:
                values[j, i] = metric * scale
        mesh = ax.pcolormesh(x_edges, y_edges, values, cmap="viridis",
                              shading="flat")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax)
    fig.suptitle(f"stage {stage}: metrics panel (worst arm per cell)")
    fig.savefig(OUT / f"stage{stage}_metrics_panel.png", dpi=200)
    plt.close(fig)


def _heatmap_and_panel(stage, rows, x_grid, y_grid, x_key, y_key, metric_key,
                        winner):
    is_rot = metric_key == "worst_arm_rot_rmse_rad"
    unit_scale = float(np.degrees(1.0)) if is_rot else 1000.0
    unit_label = "deg" if is_rot else "mm"
    metric_name = "rotation RMSE" if is_rot else "position RMSE"
    metric_slug = "rot" if is_rot else "pos"
    log_x = x_key in ("KP_POS", "KP_ROT", "DAMPING")

    n_outer, n_inner = len(x_grid), len(y_grid)
    values, disqualified = _grid_matrices(rows, n_outer, n_inner, metric_key,
                                          unit_scale)
    winner_xy = (winner["gains"][x_key], winner["gains"][y_key]) \
        if winner is not None else None
    _heatmap(x_grid, y_grid, x_key, y_key, values, disqualified, winner_xy,
              f"worst-arm {metric_name} [{unit_label}]",
              f"stage {stage}: {metric_name} over {x_key} x {y_key}", log_x,
              OUT / f"stage{stage}_{metric_slug}_heatmap.png")
    _metrics_panel(stage, rows, x_grid, y_grid, x_key, y_key, log_x)


def _scale_curves_plot(rows, winner):
    if not rows:
        return
    scales = np.array([_parse_stage3_scale(row["config_id"]) for row in rows])
    order = np.argsort(scales)
    sides = list(rows[0]["arms"])

    fig, (ax_pos, ax_rot) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, layout="constrained")
    for side in sides:
        color, style = SIDE_COLOR[side], SIDE_STYLE[side]
        pos_mm = [
            (rows[i]["arms"][side]["position_error_norm_rmse_m"] * 1000.0
             if rows[i]["arms"][side]["position_error_norm_rmse_m"] is not None
             else np.nan)
            for i in order]
        rot_deg = [
            (np.degrees(rows[i]["arms"][side]["rotation_error_norm_rmse_rad"])
             if rows[i]["arms"][side]["rotation_error_norm_rmse_rad"] is not None
             else np.nan)
            for i in order]
        ax_pos.plot(scales[order], pos_mm, color=color, linestyle=style,
                    marker="o", label=side)
        ax_rot.plot(scales[order], rot_deg, color=color, linestyle=style,
                    marker="o", label=side)
    if winner is not None:
        winner_scale = _parse_stage3_scale(winner["config_id"])
        for ax in (ax_pos, ax_rot):
            ax.axvline(winner_scale, color="0.4", linewidth=0.8, linestyle=":")
    ax_pos.set_ylabel("position RMSE [mm]")
    ax_rot.set_ylabel("rotation RMSE [deg]")
    ax_rot.set_xlabel("scale factor on the four stage-1/2 winner task gains")
    ax_pos.legend()
    fig.suptitle("stage 3: uniform scale sweep on task gains")
    fig.savefig(OUT / "stage3_scale_curves.png", dpi=200)
    plt.close(fig)


def _make_stage_plots(stage, rows, metric_key):
    OUT.mkdir(parents=True, exist_ok=True)
    winner = select_winner(rows, metric_key, strict=False)
    if stage == 1:
        _heatmap_and_panel(stage, rows, KP_POS_GRID, KD_POS_GRID,
                            "KP_POS", "KD_POS", metric_key, winner)
    elif stage == 2:
        _heatmap_and_panel(stage, rows, KP_ROT_GRID, KD_ROT_GRID,
                            "KP_ROT", "KD_ROT", metric_key, winner)
    elif stage == 3:
        _scale_curves_plot(rows, winner)
    elif stage == 4:
        _heatmap_and_panel(stage, rows, DAMPING_GRID, K_NULL_GRID,
                            "DAMPING", "K_NULL", metric_key, winner)


def _progression_point(row):
    pos = _worst_over_arms(row["arms"], "position_error_norm_rmse_m")
    rot = _worst_over_arms(row["arms"], "rotation_error_norm_rmse_rad")
    pos_mm = pos * 1000.0 if pos is not None else float("nan")
    rot_deg = float(np.degrees(rot)) if rot is not None else float("nan")
    return pos_mm, rot_deg


def _make_progression_plot(state):
    labels, pos_mm, rot_deg = [], [], []
    stage0 = state["stages"].get("0")
    if stage0 and stage0["rows"]:
        row = next(iter(stage0["rows"].values()))
        pos, rot = _progression_point(row)
        labels.append("baseline")
        pos_mm.append(pos)
        rot_deg.append(rot)
    for stage in (1, 2, 3, 4):
        stage_state = state["stages"].get(str(stage))
        if not stage_state or not stage_state.get("winner_config_id"):
            continue
        row = stage_state["rows"][stage_state["winner_config_id"]]
        pos, rot = _progression_point(row)
        labels.append(f"after\nstage {stage}")
        pos_mm.append(pos)
        rot_deg.append(rot)
    if len(labels) < 2:
        return

    fig, (ax_pos, ax_rot) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, layout="constrained")
    ax_pos.plot(labels, pos_mm, "o-", color=C_XYZ[0])
    ax_rot.plot(labels, rot_deg, "o-", color=C_XYZ[2])
    ax_pos.set_ylabel("worst-arm position RMSE [mm]")
    ax_rot.set_ylabel("worst-arm rotation RMSE [deg]")
    fig.suptitle("What staged tuning bought: baseline -> each stage's winner")
    fig.savefig(OUT / "sweep_progression.png", dpi=200)
    plt.close(fig)


# --- stage runner ----------------------------------------------------------


def run_stage(stage, state, args):
    """Run stage's remaining episodes (skipping already-completed
    config_ids), select its winner, re-run the winner once to persist
    via live.save_run, and (re)write this stage's plots + the overall
    summary.csv + progression plot. Returns the winning row, or None for
    stage 0 (no selection, just the progression-plot anchor)."""
    winners = _collected_winners(state, stage)
    grid = stage_grid(stage, winners)
    stage_state = state["stages"].setdefault(
        str(stage), {"rows": {}, "winner_config_id": None})
    if stage >= 2:
        # Recorded so a later --sat-threshold-only resume can tell
        # whether this stage's inherited base gains have since changed
        # (see _reconcile_state_for_threshold_change).
        stage_state["grid_signature"] = _stage_grid_signature(grid)
    rows_by_id = stage_state["rows"]

    stage2_rot_rmse = None
    if stage in (3, 4):
        stage2_state = state["stages"].get("2")
        if stage2_state and stage2_state.get("winner_config_id"):
            stage2_rot_rmse = stage2_state["rows"][
                stage2_state["winner_config_id"]]["worst_arm_rot_rmse_rad"]

    remaining = [(cid, gains) for cid, gains in grid if cid not in rows_by_id]
    total = len(grid)
    already_done = total - len(remaining)
    start = time.monotonic()
    for run_count, (config_id, gains) in enumerate(remaining, start=1):
        row = run_episode(gains)
        row["stage"] = stage
        row["config_id"] = config_id
        row["disqualification"] = (
            [] if stage == 0 else
            disqualify_reasons(row, stage, sat_threshold=args.sat_threshold,
                                stage2_winner_rot_rmse=stage2_rot_rmse))
        rows_by_id[config_id] = row

        elapsed = time.monotonic() - start
        per_episode = elapsed / run_count
        done = already_done + run_count
        remaining_left = total - done
        print(f"stage {stage} [{done}/{total}] {config_id}: "
              f"{per_episode:.1f} s/episode, "
              f"~{per_episode * remaining_left / 60.0:.1f} min left")
        write_state_atomic(state)

    rows = [rows_by_id[cid] for cid, _ in grid]
    _write_summary_csv(state)

    if stage == 0:
        _make_progression_plot(state)
        return None

    metric_key = "worst_arm_rot_rmse_rad" if stage == 2 else "worst_arm_pos_rmse_m"
    _make_stage_plots(stage, rows, metric_key)
    _make_progression_plot(state)

    winner = select_winner(rows, metric_key)
    stage_state["winner_config_id"] = winner["config_id"]
    write_state_atomic(state)

    winner_log = live.run_experiment(SCENARIO, gains=winner["gains"])
    live.save_run(winner_log, SCENARIO, OUT / f"stage{stage}_winner_run")
    return winner


def _regenerate_stage_outputs(stage, state):
    stage_state = state["stages"].get(str(stage))
    if not stage_state or not stage_state["rows"]:
        print(f"stage {stage}: no saved episodes in state, skipping")
        return
    winners = _collected_winners(state, stage)
    grid = stage_grid(stage, winners)
    rows = [stage_state["rows"][cid] for cid, _ in grid
            if cid in stage_state["rows"]]
    _write_summary_csv(state)
    if stage == 0:
        _make_progression_plot(state)
        return
    metric_key = "worst_arm_rot_rmse_rad" if stage == 2 else "worst_arm_pos_rmse_m"
    _make_stage_plots(stage, rows, metric_key)
    _make_progression_plot(state)


# --- CLI ---------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Staged sequential gain sweep for the reactive "
                    "controller (KP_POS/KD_POS -> KP_ROT/KD_ROT -> "
                    "task-gain scale -> DAMPING/K_NULL -> report).")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4),
                        help="run only this stage (earlier stages' winners "
                             "must already be in sweep_state.json)")
    parser.add_argument("--fresh", action="store_true",
                        help="discard saved sweep_state.json and start over")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny grids and a short scenario, written to "
                             "analysis/output/gain_sweep_smoke/ (~13 "
                             "episodes, minutes) -- pipeline smoke test")
    parser.add_argument("--plots-only", action="store_true",
                        help="regenerate plots and summary.csv from the "
                             "saved state; run no new episodes")
    parser.add_argument("--sat-threshold", type=float, default=SAT_THRESHOLD_PCT,
                        help="worst-arm velocity-saturation disqualification "
                             f"threshold, percent (default {SAT_THRESHOLD_PCT})")
    args = parser.parse_args(argv)

    if args.smoke:
        _use_smoke_mode()

    state = load_or_init_state(args)
    stages = (args.stage,) if args.stage is not None else (0, 1, 2, 3, 4)

    for stage in stages:
        if stage >= 2 and str(stage - 1) not in state["stages"]:
            raise RuntimeError(
                f"stage {stage} requires stage {stage - 1}'s winner in "
                f"{OUT}/sweep_state.json; run the earlier stage(s) first")
        if args.plots_only:
            _regenerate_stage_outputs(stage, state)
        else:
            run_stage(stage, state, args)

    write_state_atomic(state)
    print(f"Done. See {OUT}/summary.csv and {OUT}/*.png")


if __name__ == "__main__":
    main()
