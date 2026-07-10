"""Parallel KP_ROT x KD_ROT sweep for the dual-arm reactive controller."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import live
from controller import servo


OUT = Path("analysis/output/orientation_gain_sweep")
KP_ROT_GRID = [2, 12, 22, 32, 42, 52, 62, 72, 80]
KD_ROT_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
FIXED_GAIN_NAMES = ("KP_POS", "KD_POS", "K_NULL", "DAMPING")
FIXED_GAINS = {name: float(getattr(servo, name)) for name in FIXED_GAIN_NAMES}
SCENARIO = live.ExperimentConfig(
    arms=("right", "left"),
    linear_amplitude=np.array([0.18, 0.04, 0.05]),
    rotational_amplitude=np.array([0.0, 0.0, -0.2]),
    linear_frequency=0.5,
    rotational_frequency=0.5,
    evaluation_seconds=4.0,
    settle_timeout=20.0,
)
METRIC_SCHEMA = {
    "orientation_error_norm_rmse_deg": "degrees(sqrt(mean(sum(e_rot**2, axis=1))))",
    "orientation_error_norm_max_deg": "degrees(max(norm(e_rot, axis=1)))",
    "peak_measured_joint_speed_rad_s": "max(abs(qdot_measured))",
    "ee_angular_speed_norm_rmse_rad_s": "sqrt(mean(sum(e_w**2, axis=1)))",
}
METRIC_LABELS = {
    "orientation_error_norm_rmse_deg": ("orientation error norm RMSE", "deg"),
    "orientation_error_norm_max_deg": ("maximum orientation error norm", "deg"),
    "peak_measured_joint_speed_rad_s": ("peak measured joint speed", "rad/s"),
    "ee_angular_speed_norm_rmse_rad_s": ("EE angular speed norm RMSE", "rad/s"),
}
SUMMARY_FIELDS = [
    "config_id", "kp_rot", "kd_rot", "status", "settled",
    "settle_duration_s", "valid", "warning_reasons",
    "evaluation_sample_count", "error",
] + [
    f"{side}_{metric}"
    for side in SCENARIO.arms
    for metric in METRIC_SCHEMA
]
ROW_STATUSES = {"completed", "pending", "worker_error"}


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _arm_metrics(fields, mask):
    if not np.any(mask):
        return {name: None for name in METRIC_SCHEMA}
    e_rot = np.asarray(fields["e_rot"], dtype=float)[mask]
    e_w = np.asarray(fields["e_w"], dtype=float)[mask]
    qdot = np.asarray(fields["qdot_measured"], dtype=float)[mask]
    return {
        "orientation_error_norm_rmse_deg": _finite(
            np.degrees(np.sqrt(np.mean(np.sum(e_rot**2, axis=1))))),
        "orientation_error_norm_max_deg": _finite(
            np.degrees(np.max(np.linalg.norm(e_rot, axis=1)))),
        "peak_measured_joint_speed_rad_s": _finite(np.max(np.abs(qdot))),
        "ee_angular_speed_norm_rmse_rad_s": _finite(
            np.sqrt(np.mean(np.sum(e_w**2, axis=1)))),
    }


def episode_row(log, kp_rot, kd_rot):
    mask = np.asarray(log.evaluation_mask, dtype=bool)
    warnings = list(log.warning_reasons)
    arms = {side: _arm_metrics(log.arm_data[side], mask) for side in log.arms}
    if not np.any(mask):
        warnings.append("missing evaluation samples")
    if any(value is None for values in arms.values() for value in values.values()):
        if "non-finite data" not in warnings and np.any(mask):
            warnings.append("non-finite data")
    valid = bool(log.settled and np.any(mask)
                 and "non-finite data" not in warnings
                 and all(value is not None for values in arms.values()
                         for value in values.values()))
    return {
        "config_id": config_id(kp_rot, kd_rot),
        "kp_rot": float(kp_rot),
        "kd_rot": float(kd_rot),
        "status": "completed",
        "settled": bool(log.settled),
        "settle_duration_s": _finite(log.settle_duration),
        "valid": valid,
        "warning_reasons": warnings,
        "evaluation_sample_count": int(np.count_nonzero(mask)),
        "arms": arms,
    }


def config_id(kp_rot, kd_rot):
    return f"kp{float(kp_rot):g}_kd{float(kd_rot):g}"


def all_jobs():
    return [(kp, kd) for kp in KP_ROT_GRID for kd in KD_ROT_GRID]


def fingerprint_payload():
    scenario = live._config_json(SCENARIO)
    return {
        "grid": {"kp_rot": KP_ROT_GRID, "kd_rot": KD_ROT_GRID},
        "scenario": {
            key: scenario[key] for key in (
                "arms", "linear_amplitude", "linear_frequency",
                "rotational_amplitude", "rotational_frequency")
        },
        "settling": {
            key: scenario[key] for key in (
                "settle_pos_tol", "settle_rot_tol", "settle_dwell",
                "settle_timeout")
        },
        "evaluation_seconds": SCENARIO.evaluation_seconds,
        "fixed_gains": FIXED_GAINS,
        "metric_schema": METRIC_SCHEMA,
    }


def resume_fingerprint(payload=None):
    encoded = json.dumps(payload or fingerprint_payload(), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def new_state(workers=None, timestamp=None):
    workers = min(4, os.cpu_count() or 1) if workers is None else workers
    timestamp = _timestamp() if timestamp is None else timestamp
    return {
        "fingerprint": resume_fingerprint(),
        "fingerprint_payload": fingerprint_payload(),
        "results": {},
        "provenance": {
            "timestamp": timestamp,
            "worker_count": workers,
            "timestep": float(live.world.model.opt.timestep),
            "dependency_versions": _dependency_versions(),
            "git_revision": live._git_revision(),
            "execution_sessions": [
                {"timestamp": timestamp, "worker_count": workers}
            ],
        },
    }


def record_execution_session(state, workers, timestamp=None):
    state["provenance"]["execution_sessions"].append({
        "timestamp": _timestamp() if timestamp is None else timestamp,
        "worker_count": workers,
    })


def write_state_atomic(state):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_state.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
    os.replace(temporary, path)


def _malformed_state(path, detail):
    raise RuntimeError(
        f"{path} is malformed ({detail}); pass --fresh to start over")


def _valid_worker_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _finite_number_or_none(value):
    return (value is None
            or (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)))


def _validate_completed_row(identifier, row, path):
    kp_rot = row.get("kp_rot")
    kd_rot = row.get("kd_rot")
    if (not _finite_number_or_none(kp_rot) or kp_rot is None
            or not _finite_number_or_none(kd_rot) or kd_rot is None
            or kp_rot not in KP_ROT_GRID or kd_rot not in KD_ROT_GRID):
        _malformed_state(path, f"completed result {identifier!r} is off-grid")
    expected_id = config_id(kp_rot, kd_rot)
    if row.get("config_id") != expected_id or identifier != expected_id:
        _malformed_state(
            path, f"completed result {identifier!r} has inconsistent identity")
    if not isinstance(row.get("settled"), bool):
        _malformed_state(path, f"completed result {identifier!r} has invalid settled")
    if not isinstance(row.get("valid"), bool):
        _malformed_state(path, f"completed result {identifier!r} has invalid valid")
    if ("settle_duration_s" not in row
            or not _finite_number_or_none(row["settle_duration_s"])):
        _malformed_state(
            path, f"completed result {identifier!r} has invalid settle duration")
    warnings = row.get("warning_reasons")
    if (not isinstance(warnings, list)
            or not all(isinstance(reason, str) for reason in warnings)):
        _malformed_state(path, f"completed result {identifier!r} has invalid warnings")
    sample_count = row.get("evaluation_sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count < 0):
        _malformed_state(
            path, f"completed result {identifier!r} has invalid sample count")
    arms = row.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(SCENARIO.arms):
        _malformed_state(path, f"completed result {identifier!r} has invalid arms")
    for side in SCENARIO.arms:
        metrics = arms[side]
        if not isinstance(metrics, dict):
            _malformed_state(
                path, f"completed result {identifier!r} has invalid {side} metrics")
        for metric in METRIC_SCHEMA:
            if (metric not in metrics
                    or not _finite_number_or_none(metrics[metric])):
                _malformed_state(
                    path,
                    f"completed result {identifier!r} has invalid {side} metrics")


def _validate_state(state, path):
    if not isinstance(state, dict):
        _malformed_state(path, "top level must be an object")
    results = state.get("results")
    if not isinstance(results, dict):
        _malformed_state(path, "results must be an object")
    for identifier, row in results.items():
        if not isinstance(identifier, str) or not isinstance(row, dict):
            _malformed_state(path, "result entries must be named objects")
        status = row.get("status")
        if status not in ROW_STATUSES:
            _malformed_state(path, f"result {identifier!r} has invalid status")
        if status == "completed":
            _validate_completed_row(identifier, row, path)

    provenance = state.get("provenance")
    if not isinstance(provenance, dict):
        _malformed_state(path, "provenance must be an object")
    timestamp = provenance.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        _malformed_state(path, "provenance timestamp must be a string")
    if not _valid_worker_count(provenance.get("worker_count")):
        _malformed_state(path, "provenance worker_count must be positive")
    timestep = provenance.get("timestep")
    if (not isinstance(timestep, (int, float)) or isinstance(timestep, bool)
            or not np.isfinite(timestep) or timestep <= 0):
        _malformed_state(path, "provenance timestep must be positive")
    versions = provenance.get("dependency_versions")
    if (not isinstance(versions, dict)
            or not all(isinstance(name, str) and isinstance(version, str)
                       for name, version in versions.items())):
        _malformed_state(path, "dependency_versions must be a string mapping")
    revision = provenance.get("git_revision")
    if not isinstance(revision, str) or not revision:
        _malformed_state(path, "git_revision must be a string")
    sessions = provenance.get("execution_sessions")
    if not isinstance(sessions, list) or not sessions:
        _malformed_state(path, "execution_sessions must be a non-empty list")
    for session in sessions:
        if not isinstance(session, dict):
            _malformed_state(path, "execution session must be an object")
        session_timestamp = session.get("timestamp")
        if not isinstance(session_timestamp, str) or not session_timestamp:
            _malformed_state(path, "execution session timestamp must be a string")
        if not _valid_worker_count(session.get("worker_count")):
            _malformed_state(path, "execution session worker_count must be positive")


def load_or_init_state(args):
    path = OUT / "sweep_state.json"
    if args.fresh and path.exists():
        path.unlink()
    if not path.exists():
        if args.plots_only:
            raise RuntimeError("--plots-only requires a compatible saved state")
        return new_state(args.workers)
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _malformed_state(path, f"cannot read valid JSON: {error}")
    _validate_state(state, path)
    if state.get("fingerprint") != resume_fingerprint():
        raise RuntimeError(
            f"{path} does not match the current sweep; pass --fresh to start over")
    return state


def pending_jobs(state, jobs=None):
    pending = []
    for job in jobs or all_jobs():
        row = state["results"].get(config_id(*job))
        if row is None:
            pending.append(job)
            continue
        status = row.get("status")
        if status not in ROW_STATUSES:
            raise RuntimeError(f"result {config_id(*job)!r} has invalid status")
        if status != "completed":
            pending.append(job)
    return pending


def run_episode(job):
    kp_rot, kd_rot = job
    gains = dict(FIXED_GAINS, KP_ROT=float(kp_rot), KD_ROT=float(kd_rot))
    log = live.run_experiment(SCENARIO, gains=gains)
    return episode_row(log, kp_rot, kd_rot)


def run_pending(state, jobs, workers):
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(run_episode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "config_id": config_id(*job),
                    "kp_rot": float(job[0]),
                    "kd_rot": float(job[1]),
                    "status": "worker_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            state["results"][config_id(*job)] = row
            write_state_atomic(state)
    return state


def heatmap_matrix(rows, side, metric, kp_grid=None, kd_grid=None):
    kp_grid = KP_ROT_GRID if kp_grid is None else kp_grid
    kd_grid = KD_ROT_GRID if kd_grid is None else kd_grid
    values = np.full((len(kd_grid), len(kp_grid)), np.nan)
    invalid = np.ones(values.shape, dtype=bool)
    kp_index = {float(value): index for index, value in enumerate(kp_grid)}
    kd_index = {float(value): index for index, value in enumerate(kd_grid)}
    for row in rows:
        if row.get("status") != "completed":
            continue
        i = kp_index[float(row["kp_rot"])]
        j = kd_index[float(row["kd_rot"])]
        value = row.get("arms", {}).get(side, {}).get(metric)
        if value is not None:
            values[j, i] = float(value)
        invalid[j, i] = not bool(row.get("valid")) or value is None
    return values, invalid


def _write_heatmap(rows, side, metric):
    values, invalid = heatmap_matrix(rows, side, metric)
    masked = np.ma.masked_where(invalid, values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    for j in range(len(KD_ROT_GRID)):
        for i in range(len(KP_ROT_GRID)):
            label = "invalid" if invalid[j, i] else f"{values[j, i]:.3g}"
            ax.text(i, j, label, ha="center", va="center", fontsize=6,
                    color="black" if invalid[j, i] else "white")
    ax.set_xticks(range(len(KP_ROT_GRID)), labels=KP_ROT_GRID)
    ax.set_yticks(range(len(KD_ROT_GRID)), labels=KD_ROT_GRID)
    ax.set_xlabel("KP_ROT [1/s]")
    ax.set_ylabel("KD_ROT [dimensionless]")
    title, unit = METRIC_LABELS[metric]
    ax.set_title(f"{side} arm: {title} [{unit}]")
    fig.colorbar(image, ax=ax, label=f"{title} [{unit}]")
    fig.savefig(OUT / f"{side}_{metric}_heatmap.png", dpi=200)
    plt.close(fig)


def write_summary(state):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for kp_rot, kd_rot in all_jobs():
        identifier = config_id(kp_rot, kd_rot)
        rows.append(state["results"].get(identifier, {
            "config_id": identifier,
            "kp_rot": float(kp_rot),
            "kd_rot": float(kd_rot),
            "status": "pending",
        }))
    flat = []
    for row in rows:
        item = {field: "" for field in SUMMARY_FIELDS}
        item.update({key: value for key, value in row.items()
                     if key in SUMMARY_FIELDS})
        item["warning_reasons"] = "; ".join(row.get("warning_reasons", []))
        for side in SCENARIO.arms:
            for metric in METRIC_SCHEMA:
                value = row.get("arms", {}).get(side, {}).get(metric)
                item[f"{side}_{metric}"] = "" if value is None else value
        flat.append(item)
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(flat)
    return rows


def _dependency_versions():
    versions = {}
    for package in ("numpy", "matplotlib", "mujoco", "pin"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def write_metadata(state):
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = state["provenance"]
    metadata = {
        **fingerprint_payload(),
        "units": {
            "KP_ROT": "1/s", "KD_ROT": "dimensionless",
            **{key: unit for key, (_, unit) in METRIC_LABELS.items()},
        },
        **provenance,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))


def write_artifacts(state):
    rows = write_summary(state)
    write_metadata(state)
    for side in SCENARIO.arms:
        for metric in METRIC_SCHEMA:
            _write_heatmap(rows, side, metric)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int,
                        default=min(4, os.cpu_count() or 1))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--plots-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    state = load_or_init_state(args)
    jobs = pending_jobs(state)
    if not args.plots_only and jobs:
        if state["results"]:
            record_execution_session(state, args.workers)
        run_pending(state, jobs, args.workers)
    write_artifacts(state)


if __name__ == "__main__":
    main()
