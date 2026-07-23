"""Reproducible settling and evaluation runs for controller diagnostics."""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from analysis import metrics, policy, provenance
from controller import desired_pos, frames, reactive_controller, servo
from controller.runner import ReactivePositionRunner
from runtime_config import (
    CONFIG,
    control_with_legacy_overrides,
    legacy_gain_dict,
)
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
    traces: servo.DualArmControlTraces
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
    metrics_computable: bool
    accepted: bool
    contact_observed: bool
    joint_limit_within_tolerance: bool
    limit_penetration_rad: float
    warning_reasons: tuple[str, ...]

    @property
    def evaluation_mask(self):
        return self.phase == "evaluation"

    @property
    def valid(self):
        """Compatibility alias. New code must use ``accepted`` explicitly."""
        return self.accepted


_TRACE_FIELDS = tuple(servo.ControlTrace.__dataclass_fields__)
_EXECUTION_CONTRACT = {
    "runner": "ReactivePositionRunner",
    "controller_pipeline": "ReactivePositionPipeline",
    "backend": "MujocoBackend",
    "cycle_operation": "exchange",
    "timing_source": "MuJoCo data.time",
    "command": "dual-arm joint position, rad",
}


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
    world.backend.release()
    world.backend.configure_torso_driver(None, None)
    world.backend.reset()
    return desired_pos.apply()


class _ExperimentTorsoDriver:
    """Explicit scenario phase; MuJoCo still owns all writes and stepping."""

    def __init__(self, scenario):
        self._scenario = scenario
        self._evaluation_start_s = None

    def start_evaluation(self, sample_time_s):
        if self._evaluation_start_s is not None:
            raise RuntimeError("evaluation has already started")
        self._evaluation_start_s = float(sample_time_s)

    def pose_at(self, sample_time_s):
        if self._evaluation_start_s is None:
            return motion.HOME_POS.copy(), motion.HOME_RPY.copy()
        return motion.torso_pose_at(
            sample_time_s - self._evaluation_start_s,
            **self._scenario,
        )

    def twist_at(self, sample_time_s):
        if self._evaluation_start_s is None:
            return np.zeros(3), np.zeros(3)
        return motion.torso_twist_at(
            sample_time_s - self._evaluation_start_s,
            **self._scenario,
        )


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
    def __init__(self, arms, control):
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
        self.gain_snapshots = [legacy_gain_dict(control)]

    def append(self, phase, sim_time, contact_time, eval_time, base_disp,
               base_v, base_w, traces, pairs, on_update):
        segment = 0
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
        status = policy.assess_run(
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
            metrics_computable=status.metrics_computable,
            accepted=status.accepted,
            contact_observed=status.contact_observed,
            joint_limit_within_tolerance=(
                status.joint_limit_within_tolerance),
            limit_penetration_rad=status.limit_penetration_rad,
            warning_reasons=status.warning_reasons,
            **arrays,
        )


def _advance(builder, runner, phase, eval_time, on_update):
    cycle = runner.cycle()
    plant = cycle.input_state
    sim_time = plant.sample_time_s
    base_v = plant.torso_twist_world.linear_m_s
    base_w = plant.torso_twist_world.angular_rad_s
    base_disp = (
        plant.torso_pose_world.position_m - motion.HOME_POS
        if phase == "evaluation"
        else np.zeros(3)
    )
    contact_time = sim_time
    pairs = _contact_pairs()
    builder.append(phase, sim_time, contact_time, eval_time, base_disp,
                   base_v, base_w, cycle.traces, pairs, on_update)
    return cycle


def _current_pose_within(runner, source_targets, config):
    plant = runner.current_state
    resolved = frames.resolve_targets_world(
        plant, world.MOUNT_CALIBRATION, source_targets)
    states = frames.controller_states(
        plant, world.MOUNT_CALIBRATION)
    for side in config.arms:
        e_pos, e_rot = reactive_controller.pose_error(
            states.for_arm(side), resolved.for_arm(side))
        if (
            np.linalg.norm(e_pos) > config.settle_pos_tol
            or np.linalg.norm(e_rot) > config.settle_rot_tol
        ):
            return False
    return True


def run_experiment(config, on_update=None, gains=None):
    """Run one settle+evaluation experiment. gains, if given, reconstructs
    an immutable controller configuration for this run before the log builder
    takes its first gain snapshot. Not an
    ExperimentConfig field: the config's exact fields are contract-tested
    (tests/test_live.py) and serialized whole in every run's
    metadata.json; gains are already persisted via gain snapshots."""
    _validate_config(config)
    config = config._replace(
        linear_amplitude=np.asarray(config.linear_amplitude, dtype=float).copy(),
        rotational_amplitude=np.asarray(
            config.rotational_amplitude, dtype=float).copy(),
    )
    scenario = {
        "linear_amplitude": config.linear_amplitude,
        "linear_frequency": config.linear_frequency,
        "rotational_amplitude": config.rotational_amplitude,
        "rotational_frequency": config.rotational_frequency,
    }
    control = control_with_legacy_overrides(CONFIG.reactive_pose, gains)
    source_targets = _reset_simulation()
    driver = _ExperimentTorsoDriver(scenario)
    world.backend.configure_torso_driver(driver.pose_at, driver.twist_at)
    runner = ReactivePositionRunner(
        world.backend,
        world.MOUNT_CALIBRATION,
        world.PIPELINE_SETUP,
        source_targets,
        config.arms,
        control,
    )
    runner.start()
    try:
        builder = _LogBuilder(tuple(config.arms), control)
        dt = runner.current_state.nominal_dt_s
        settled = False
        dwell_start = None
        settle_start = runner.current_state.sample_time_s
        while (
            runner.current_state.sample_time_s - settle_start
            < config.settle_timeout
        ):
            within = _current_pose_within(
                runner, source_targets, config)
            next_time = (
                runner.current_state.sample_time_s + dt)
            if within:
                next_dwell_start = (
                    next_time if dwell_start is None else dwell_start)
                will_settle = (
                    next_time - next_dwell_start
                    >= config.settle_dwell
                )
            else:
                next_dwell_start = None
                will_settle = False
            will_timeout = (
                next_time - settle_start >= config.settle_timeout)
            if will_settle or will_timeout:
                driver.start_evaluation(next_time)

            _advance(
                builder, runner, "settling", -1.0, on_update)
            dwell_start = next_dwell_start
            if will_settle:
                settled = True
                break

        settle_duration = (
            runner.current_state.sample_time_s - settle_start)
        evaluation_steps = max(
            1, int(np.ceil(config.evaluation_seconds / dt)))
        for step in range(evaluation_steps):
            _advance(
                builder,
                runner,
                "evaluation",
                step * dt,
                on_update,
            )
        return builder.build(config, settled, settle_duration)
    finally:
        runner.close()
        world.backend.configure_torso_driver(None, None)


def _git_revision():
    return provenance.git_provenance()["revision"]


def _asset_hash():
    return provenance._scene_asset_sha256()


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


def _metadata(log, revision, identity=None, git=None):
    identity = (provenance.experiment_identity(
        _config_json(log.config), log.gain_snapshots[0])
        if identity is None else identity)
    git = provenance.git_provenance() if git is None else git
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
        "metrics_computable": log.metrics_computable,
        "accepted": log.accepted,
        "contact_observed": log.contact_observed,
        "joint_limit_within_tolerance": log.joint_limit_within_tolerance,
        "limit_penetration_rad": log.limit_penetration_rad,
        "valid": log.accepted,
        "warning_reasons": list(log.warning_reasons),
        "acceptance_policy_version": policy.ACCEPTANCE_POLICY_VERSION,
        "evidence_schema_version": policy.EVIDENCE_SCHEMA_VERSION,
        "joint_limit_max_penetration_rad": (
            policy.JOINT_LIMIT_MAX_PENETRATION_RAD),
        "dependency_versions": identity["environment"]["dependencies"],
        "environment": identity["environment"],
        "git_revision": revision,
        "git": git,
        "source_sha256": identity["source_sha256"],
        "analysis_sha256": identity["analysis_sha256"],
        "experiment_identity_sha256": identity["identity_sha256"],
        "scene_asset_sha256": _asset_hash(),
        "controller_configuration": identity["controller"],
        "execution": dict(_EXECUTION_CONTRACT),
        "effective_control_config": identity["effective_control_config"],
        "control_config_sha256": identity["control_config_sha256"],
        "arms": list(log.arms),
    }


def save_run(log, config, output_root, *, canonical=False, final_outputs=()):
    output_root = Path(output_root)
    _validate_config(config)
    normalized_config = config._replace(
        linear_amplitude=np.asarray(config.linear_amplitude, dtype=float),
        rotational_amplitude=np.asarray(config.rotational_amplitude, dtype=float),
    )
    if not _configs_equal(log.config, normalized_config):
        raise ValueError("config does not match the configuration used for log")
    git = provenance.git_provenance()
    environment = provenance.environment_snapshot()
    if canonical:
        provenance.require_canonical_preconditions(log, git, environment)
    run_metrics = metrics.experiment_metrics(log)
    flat_metrics = metrics.flatten_metrics(run_metrics)
    revision = git["revision"]
    identity = provenance.experiment_identity(
        _config_json(log.config), log.gain_snapshots[0])
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
        json.dumps(_metadata(log, revision, identity, git), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, allow_nan=False) + "\n")
    with (run_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_metrics)
        writer.writeheader()
        writer.writerow(flat_metrics)
    artifact_digests, output_digests = provenance.artifact_hashes(
        run_dir, final_outputs)
    manifest = {
        "evidence_schema_version": policy.EVIDENCE_SCHEMA_VERSION,
        "classification": "canonical" if canonical else "exploratory",
        "accepted": log.accepted,
        "experiment_identity_sha256": identity["identity_sha256"],
        "source_sha256": identity["source_sha256"],
        "analysis_sha256": identity["analysis_sha256"],
        "scene_asset_sha256": identity["scene_asset_sha256"],
        "git": git,
        "environment": environment,
        "configuration": _config_json(log.config),
        "controller_configuration": identity["controller"],
        "execution": dict(_EXECUTION_CONTRACT),
        "effective_control_config": identity["effective_control_config"],
        "control_config_sha256": identity["control_config_sha256"],
        "policy": {
            "acceptance_policy_version": policy.ACCEPTANCE_POLICY_VERSION,
            "joint_limit_max_penetration_rad": (
                policy.JOINT_LIMIT_MAX_PENETRATION_RAD),
        },
        "artifacts": artifact_digests,
        "final_outputs": output_digests,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
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
            metrics_computable=bool(metadata.get(
                "metrics_computable", metadata["valid"])),
            accepted=bool(metadata.get("accepted", metadata["valid"])),
            contact_observed=bool(metadata.get("contact_observed", False)),
            joint_limit_within_tolerance=bool(metadata.get(
                "joint_limit_within_tolerance", metadata["valid"])),
            limit_penetration_rad=float(metadata.get(
                "limit_penetration_rad", 0.0)),
            warning_reasons=tuple(metadata["warning_reasons"]),
        )
