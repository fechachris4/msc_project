"""Reproducible settling and evaluation runs for controller diagnostics."""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import NamedTuple

import mujoco
import numpy as np

from analysis import metrics
from controller import desired_pos, frames, servo
from sim import motion, world


class ExperimentConfig(NamedTuple):
    arms: tuple[str, ...]
    linear_amplitude: np.ndarray
    linear_frequency: float
    rotational_amplitude: np.ndarray
    rotational_frequency: float
    evaluation_seconds: float
    settle_pos_tol: float = 0.002
    settle_rot_tol: float = np.deg2rad(0.5)
    settle_dwell: float = 0.5
    settle_timeout: float = 15.0


@dataclass(frozen=True)
class ExperimentUpdate:
    phase: str
    sim_time: float
    contact_time: float
    eval_time: float
    gain_segment: int
    base_displacement: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    traces: dict[str, servo.ControlTrace]
    joint_margin: dict[str, np.ndarray]
    contact_count: int
    torso_contact: dict[str, bool]


@dataclass(frozen=True)
class ExperimentLog:
    arms: tuple[str, ...]
    config: ExperimentConfig
    sim_time: np.ndarray
    contact_time: np.ndarray
    eval_time: np.ndarray
    phase: np.ndarray
    gain_segment: np.ndarray
    base_displacement: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    arm_data: dict[str, dict[str, np.ndarray]]
    contact_count: np.ndarray
    torso_contact: dict[str, np.ndarray]
    contact_pairs: tuple[tuple[str, ...], ...]
    gain_snapshots: tuple[dict[str, float], ...]
    settled: bool
    settle_duration: float
    valid: bool
    warning_reasons: tuple[str, ...]

    @property
    def evaluation_mask(self):
        return self.phase == "evaluation"


_GAIN_NAMES = ("KP_POS", "KP_ROT", "KD_POS", "KD_ROT", "K_NULL", "DAMPING")
_INITIAL_GAINS = {name: float(getattr(servo, name)) for name in _GAIN_NAMES}
_TRACE_FIELDS = tuple(servo.ControlTrace.__dataclass_fields__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gain_snapshot():
    return {name: float(getattr(servo, name)) for name in _GAIN_NAMES}


def _restore_initial_gains():
    for name, value in _INITIAL_GAINS.items():
        if name == "K_NULL":
            servo.set_k_null(value)
        else:
            setattr(servo, name, value)


def _apply_gain_overrides(gains):
    """Override servo module gains for one run. K_NULL is routed through
    servo.set_k_null() (required -- it rescales _K_NULL_VEC; a plain
    setattr would silently not apply); the rest are plain setattr, same
    as _restore_initial_gains above."""
    for name, value in gains.items():
        if name not in _GAIN_NAMES:
            raise ValueError(f"unknown gain override: {name!r}")
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"gain override {name} must be finite and non-negative")
        if name == "K_NULL":
            servo.set_k_null(value)
        else:
            setattr(servo, name, value)


def _validate_config(config):
    if not config.arms or any(side not in world.SIDES for side in config.arms):
        raise ValueError(f"arms must be a non-empty subset of {world.SIDES}")
    if len(set(config.arms)) != len(config.arms):
        raise ValueError("arms must not contain duplicates")
    for name in ("linear_amplitude", "rotational_amplitude"):
        value = np.asarray(getattr(config, name), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite shape-(3,) array")
    for name in ("linear_frequency", "rotational_frequency"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(config.evaluation_seconds) or config.evaluation_seconds <= 0.0:
        raise ValueError("evaluation_seconds must be finite and positive")
    for name in ("settle_pos_tol", "settle_rot_tol", "settle_dwell",
                 "settle_timeout"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _reset_simulation():
    _restore_initial_gains()
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    servo.init_ctrl()
    mujoco.mj_forward(world.model, world.data)


def _joint_margin(side, q):
    low, high, limited = world.jnt_range(side)
    margin = np.full(7, np.inf)
    margin[limited] = np.minimum(q[limited] - low[limited],
                                 high[limited] - q[limited])
    return margin


def _contact_pairs():
    pairs = []
    for contact in world.data.contact[:world.data.ncon]:
        names = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = world.model.geom_bodyid[geom_id]
            geom = mujoco.mj_id2name(
                world.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            body = mujoco.mj_id2name(
                world.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
            names.append(f"{body or '<world>'}/{geom or f'geom_{geom_id}'}")
        pairs.append(" <> ".join(names))
    return tuple(pairs)


def _torso_contact_from_pairs(pairs, arms):
    result = {side: False for side in arms}
    for pair in pairs:
        endpoints = pair.split(" <> ")
        bodies = [endpoint.split("/", 1)[0] for endpoint in endpoints]
        if "torso" not in bodies:
            continue
        for side in arms:
            if any(body.startswith(f"{side}_") for body in bodies):
                result[side] = True
    return result


class _LogBuilder:
    def __init__(self, arms):
        self.arms = arms
        self.sim_time = []
        self.contact_time = []
        self.eval_time = []
        self.phase = []
        self.gain_segment = []
        self.base_displacement = []
        self.base_linear_velocity = []
        self.base_angular_velocity = []
        self.arm_data = {
            side: {name: [] for name in (*_TRACE_FIELDS, "joint_margin")}
            for side in arms
        }
        self.contact_count = []
        self.torso_contact = {side: [] for side in arms}
        self.contact_pairs = []
        self.gain_snapshots = [_gain_snapshot()]

    def append(self, phase, sim_time, contact_time, eval_time, base_disp,
               base_v, base_w, traces, pairs, on_update):
        snapshot = _gain_snapshot()
        if snapshot != self.gain_snapshots[-1]:
            self.gain_snapshots.append(snapshot)
        segment = len(self.gain_snapshots) - 1
        margins = {}
        for side in self.arms:
            trace = traces[side]
            for name in _TRACE_FIELDS:
                self.arm_data[side][name].append(np.asarray(getattr(trace, name)))
            margins[side] = _joint_margin(side, trace.q)
            self.arm_data[side]["joint_margin"].append(margins[side])
        torso = _torso_contact_from_pairs(pairs, self.arms)
        self.sim_time.append(float(sim_time))
        self.contact_time.append(float(contact_time))
        self.eval_time.append(float(eval_time))
        self.phase.append(phase)
        self.gain_segment.append(segment)
        self.base_displacement.append(np.asarray(base_disp, dtype=float).copy())
        self.base_linear_velocity.append(np.asarray(base_v, dtype=float).copy())
        self.base_angular_velocity.append(np.asarray(base_w, dtype=float).copy())
        self.contact_count.append(len(pairs))
        self.contact_pairs.append(tuple(pairs))
        for side in self.arms:
            self.torso_contact[side].append(torso[side])
        if on_update is not None:
            on_update(ExperimentUpdate(
                phase=phase,
                sim_time=float(sim_time),
                contact_time=float(contact_time),
                eval_time=float(eval_time),
                gain_segment=segment,
                base_displacement=np.asarray(base_disp, dtype=float).copy(),
                base_linear_velocity=np.asarray(base_v, dtype=float).copy(),
                base_angular_velocity=np.asarray(base_w, dtype=float).copy(),
                traces=traces,
                joint_margin={side: value.copy() for side, value in margins.items()},
                contact_count=len(pairs),
                torso_contact=torso,
            ))

    def build(self, config, settled, settle_duration):
        arrays = {
            "sim_time": np.asarray(self.sim_time, dtype=float),
            "contact_time": np.asarray(self.contact_time, dtype=float),
            "eval_time": np.asarray(self.eval_time, dtype=float),
            "phase": np.asarray(self.phase, dtype="U10"),
            "gain_segment": np.asarray(self.gain_segment, dtype=int),
            "base_displacement": np.asarray(self.base_displacement, dtype=float),
            "base_linear_velocity": np.asarray(self.base_linear_velocity, dtype=float),
            "base_angular_velocity": np.asarray(self.base_angular_velocity, dtype=float),
            "contact_count": np.asarray(self.contact_count, dtype=int),
        }
        arm_data = {
            side: {name: np.asarray(values) for name, values in fields.items()}
            for side, fields in self.arm_data.items()
        }
        torso_contact = {
            side: np.asarray(values, dtype=bool)
            for side, values in self.torso_contact.items()
        }
        reasons = _validity_reasons(
            config, arrays, arm_data, torso_contact, settled)
        return ExperimentLog(
            arms=self.arms,
            config=config,
            arm_data=arm_data,
            torso_contact=torso_contact,
            contact_pairs=tuple(self.contact_pairs),
            gain_snapshots=tuple(self.gain_snapshots),
            settled=settled,
            settle_duration=float(settle_duration),
            valid=not reasons,
            warning_reasons=tuple(reasons),
            **arrays,
        )


def _validity_reasons(config, arrays, arm_data, torso_contact, settled):
    reasons = []
    if not settled:
        reasons.append("settling timeout")
    if np.any(arrays["contact_count"] > 0):
        reasons.append("contact detected")
    if any(np.any(values) for values in torso_contact.values()):
        reasons.append("torso contact detected")
    if any(np.any(fields["joint_margin"] < 0.0) for fields in arm_data.values()):
        reasons.append("negative joint margin")
    numeric = [arrays["sim_time"], arrays["contact_time"],
               arrays["eval_time"], arrays["base_displacement"],
               arrays["base_linear_velocity"],
               arrays["base_angular_velocity"]]
    for fields in arm_data.values():
        numeric.extend(value for name, value in fields.items()
                       if name != "joint_margin")
        margins = fields["joint_margin"]
        numeric.append(margins[np.isfinite(margins)])
    if any(not np.all(np.isfinite(value)) for value in numeric):
        reasons.append("non-finite data")
    periods = []
    if np.any(config.linear_amplitude) and config.linear_frequency > 0.0:
        periods.append(1.0 / config.linear_frequency)
    if np.any(config.rotational_amplitude) and config.rotational_frequency > 0.0:
        periods.append(1.0 / config.rotational_frequency)
    if periods and config.evaluation_seconds < max(periods):
        reasons.append("evaluation shorter than one disturbance period")
    return reasons


def _advance(builder, phase, eval_time, scenario, on_update):
    sim_time = float(world.data.time)
    if phase == "evaluation":
        motion.set_torso_pose(eval_time, **scenario)
        mujoco.mj_kinematics(world.model, world.data)
        base_v, base_w = motion.torso_twist_at(eval_time, **scenario)
        base_pos, _ = motion.torso_pose_at(eval_time, **scenario)
        base_disp = base_pos - motion.HOME_POS
    else:
        base_v = np.zeros(3)
        base_w = np.zeros(3)
        base_disp = np.zeros(3)
    traces = servo.apply_ctrl(
        world.model.opt.timestep, (base_v, base_w), builder.arms)
    mujoco.mj_step(world.model, world.data)
    contact_time = sim_time
    pairs = _contact_pairs()
    builder.append(phase, sim_time, contact_time, eval_time, base_disp,
                   base_v, base_w, traces, pairs, on_update)
    return traces


def run_experiment(config, on_update=None, gains=None):
    """Run one settle+evaluation experiment. gains, if given, overrides
    servo module gains (see _apply_gain_overrides) after _reset_simulation
    restores the initial gains and before the log builder takes its first
    gain snapshot, so the override self-documents in the saved run's
    metadata rather than looking like a mid-run gain change. Not an
    ExperimentConfig field: the config's exact fields are contract-tested
    (tests/test_live.py) and serialized whole in every run's
    metadata.json; gains are already persisted via gain snapshots."""
    _validate_config(config)
    config = config._replace(
        linear_amplitude=np.asarray(config.linear_amplitude, dtype=float).copy(),
        rotational_amplitude=np.asarray(
            config.rotational_amplitude, dtype=float).copy(),
    )
    _reset_simulation()
    if gains is not None:
        _apply_gain_overrides(gains)
    builder = _LogBuilder(tuple(config.arms))
    dt = float(world.model.opt.timestep)
    settled = False
    dwell_start = None
    settle_start = float(world.data.time)
    while world.data.time - settle_start < config.settle_timeout:
        traces = _advance(builder, "settling", -1.0, {}, on_update)
        within = all(
            np.linalg.norm(trace.e_pos) <= config.settle_pos_tol
            and np.linalg.norm(trace.e_rot) <= config.settle_rot_tol
            for trace in traces.values()
        )
        if within:
            if dwell_start is None:
                dwell_start = float(world.data.time)
            if world.data.time - dwell_start >= config.settle_dwell:
                settled = True
                break
        else:
            dwell_start = None
    settle_duration = float(world.data.time - settle_start)
    scenario = {
        "linear_amplitude": config.linear_amplitude,
        "linear_frequency": config.linear_frequency,
        "rotational_amplitude": config.rotational_amplitude,
        "rotational_frequency": config.rotational_frequency,
    }
    evaluation_steps = max(1, int(np.ceil(config.evaluation_seconds / dt)))
    for step in range(evaluation_steps):
        _advance(builder, "evaluation", step * dt, scenario, on_update)
    return builder.build(config, settled, settle_duration)


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _asset_hash():
    digest = hashlib.sha256()
    paths = [_PROJECT_ROOT / "sim" / "scene.xml"]
    paths.extend(sorted((_PROJECT_ROOT / "sim" / "assets").rglob("*")))
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(_PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_json(config):
    return {
        name: (np.asarray(value).tolist() if isinstance(value, np.ndarray)
               else list(value) if name == "arms" else value)
        for name, value in config._asdict().items()
    }


def _config_from_json(values):
    return ExperimentConfig(
        arms=tuple(values["arms"]),
        linear_amplitude=np.asarray(values["linear_amplitude"], dtype=float),
        linear_frequency=float(values["linear_frequency"]),
        rotational_amplitude=np.asarray(
            values["rotational_amplitude"], dtype=float),
        rotational_frequency=float(values["rotational_frequency"]),
        evaluation_seconds=float(values["evaluation_seconds"]),
        settle_pos_tol=float(values["settle_pos_tol"]),
        settle_rot_tol=float(values["settle_rot_tol"]),
        settle_dwell=float(values["settle_dwell"]),
        settle_timeout=float(values["settle_timeout"]),
    )


def _configs_equal(left, right):
    return (
        left.arms == right.arms
        and np.array_equal(left.linear_amplitude, right.linear_amplitude)
        and left.linear_frequency == right.linear_frequency
        and np.array_equal(left.rotational_amplitude,
                           right.rotational_amplitude)
        and left.rotational_frequency == right.rotational_frequency
        and left.evaluation_seconds == right.evaluation_seconds
        and left.settle_pos_tol == right.settle_pos_tol
        and left.settle_rot_tol == right.settle_rot_tol
        and left.settle_dwell == right.settle_dwell
        and left.settle_timeout == right.settle_timeout
    )


def _metadata(log, revision):
    versions = {}
    for package in ("numpy", "matplotlib", "mujoco", "pin"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {
        "configuration": _config_json(log.config),
        "gains": log.gain_snapshots[0],
        "gain_segments": list(log.gain_snapshots),
        "units": {
            "position": "m", "orientation": "rad", "time": "s",
            "linear_velocity": "m/s", "angular_velocity": "rad/s",
            "joint_position": "rad", "joint_velocity": "rad/s",
        },
        "timestep": float(world.model.opt.timestep),
        "settled": log.settled,
        "settle_duration": log.settle_duration,
        "evaluation_mask": log.evaluation_mask.tolist(),
        "valid": log.valid,
        "warning_reasons": list(log.warning_reasons),
        "dependency_versions": versions,
        "git_revision": revision,
        "scene_asset_sha256": _asset_hash(),
        "arms": list(log.arms),
    }


def save_run(log, config, output_root):
    output_root = Path(output_root)
    _validate_config(config)
    normalized_config = config._replace(
        linear_amplitude=np.asarray(config.linear_amplitude, dtype=float),
        rotational_amplitude=np.asarray(config.rotational_amplitude, dtype=float),
    )
    if not _configs_equal(log.config, normalized_config):
        raise ValueError("config does not match the configuration used for log")
    run_metrics = metrics.experiment_metrics(log)
    flat_metrics = metrics.flatten_metrics(run_metrics)
    revision = _git_revision()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / f"{stamp}-{revision[:7]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    arrays = {
        "sim_time": log.sim_time,
        "contact_time": log.contact_time,
        "eval_time": log.eval_time,
        "phase": log.phase,
        "gain_segment": log.gain_segment,
        "base_displacement": log.base_displacement,
        "base_linear_velocity": log.base_linear_velocity,
        "base_angular_velocity": log.base_angular_velocity,
        "contact_count": log.contact_count,
        "contact_pairs_json": np.asarray(
            [json.dumps(list(pairs)) for pairs in log.contact_pairs]),
    }
    for side in log.arms:
        arrays[f"torso_contact__{side}"] = log.torso_contact[side]
        for name, value in log.arm_data[side].items():
            arrays[f"arm__{side}__{name}"] = value
    np.savez_compressed(run_dir / "run.npz", **arrays)
    (run_dir / "metadata.json").write_text(
        json.dumps(_metadata(log, revision), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, allow_nan=False) + "\n")
    with (run_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_metrics)
        writer.writeheader()
        writer.writerow(flat_metrics)
    return run_dir


def load_run(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    with np.load(run_dir / "run.npz", allow_pickle=False) as stored:
        arms = tuple(metadata["arms"])
        config = _config_from_json(metadata["configuration"])
        arm_data = {
            side: {
                key.split("__", 2)[2]: stored[key].copy()
                for key in stored.files if key.startswith(f"arm__{side}__")
            }
            for side in arms
        }
        return ExperimentLog(
            arms=arms,
            config=config,
            sim_time=stored["sim_time"].copy(),
            contact_time=stored["contact_time"].copy(),
            eval_time=stored["eval_time"].copy(),
            phase=stored["phase"].copy(),
            gain_segment=stored["gain_segment"].copy(),
            base_displacement=stored["base_displacement"].copy(),
            base_linear_velocity=stored["base_linear_velocity"].copy(),
            base_angular_velocity=stored["base_angular_velocity"].copy(),
            arm_data=arm_data,
            contact_count=stored["contact_count"].copy(),
            torso_contact={
                side: stored[f"torso_contact__{side}"].copy()
                for side in arms
            },
            contact_pairs=tuple(
                tuple(json.loads(value))
                for value in stored["contact_pairs_json"].tolist()),
            gain_snapshots=tuple(dict(values)
                                 for values in metadata["gain_segments"]),
            settled=bool(metadata["settled"]),
            settle_duration=float(metadata["settle_duration"]),
            valid=bool(metadata["valid"]),
            warning_reasons=tuple(metadata["warning_reasons"]),
        )
