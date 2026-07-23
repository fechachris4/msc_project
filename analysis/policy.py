"""Authoritative experiment and timing policy for the reactive baseline."""

from dataclasses import dataclass

import numpy as np


ACCEPTANCE_POLICY_VERSION = "reactive-run-1"
EVIDENCE_SCHEMA_VERSION = "reactive-evidence-1"
JOINT_LIMIT_MAX_PENETRATION_RAD = 0.02

CONTROLLER_COMPUTE_BUDGET_S = 0.001
TELEMETRY_CAPACITY = 4096
TELEMETRY_OVERFLOW_POLICY = "drop_newest"

PRIMARY_ENVIRONMENT = {
    "python": "3.14.4",
    "numpy": "2.4.4",
    "mujoco": "3.10.0",
    "pinocchio": "4.0.0",
    "matplotlib": "3.11.0",
}


@dataclass(frozen=True)
class RunStatus:
    metrics_computable: bool
    accepted: bool
    contact_observed: bool
    joint_limit_within_tolerance: bool
    limit_penetration_rad: float
    warning_reasons: tuple[str, ...]


def assess_run(config, arrays, arm_data, torso_contact, settled):
    """Apply the versioned run policy without calculating performance metrics."""
    reasons = []
    evaluation_mask = np.asarray(arrays["phase"]) == "evaluation"
    has_evaluation = bool(np.any(evaluation_mask))
    if not has_evaluation:
        reasons.append("missing evaluation samples")

    contact_observed = bool(
        np.any(np.asarray(arrays["contact_count"]) > 0)
        or any(np.any(np.asarray(values, dtype=bool))
               for values in torso_contact.values())
    )
    if np.any(np.asarray(arrays["contact_count"]) > 0):
        reasons.append("contact detected")
    if any(np.any(np.asarray(values, dtype=bool))
           for values in torso_contact.values()):
        reasons.append("torso contact detected")

    finite_margins = []
    for fields in arm_data.values():
        margins = np.asarray(fields["joint_margin"], dtype=float)
        finite_margins.extend(margins[np.isfinite(margins)].tolist())
    minimum_margin = min(finite_margins, default=0.0)
    limit_penetration = max(0.0, -float(minimum_margin))
    joint_limit_within_tolerance = (
        limit_penetration <= JOINT_LIMIT_MAX_PENETRATION_RAD)
    if limit_penetration > 0.0:
        reasons.append("negative joint margin")
    if not joint_limit_within_tolerance:
        reasons.append("joint limit penetration exceeds tolerance")

    numeric = [
        np.asarray(arrays["sim_time"]),
        np.asarray(arrays["contact_time"]),
        np.asarray(arrays["eval_time"]),
        np.asarray(arrays["base_displacement"]),
        np.asarray(arrays["base_linear_velocity"]),
        np.asarray(arrays["base_angular_velocity"]),
    ]
    for fields in arm_data.values():
        numeric.extend(
            np.asarray(value)[evaluation_mask]
            for name, value in fields.items()
            if name != "joint_margin"
        )
    numeric_finite = all(np.all(np.isfinite(value)) for value in numeric)
    metrics_computable = has_evaluation and numeric_finite
    if not numeric_finite:
        reasons.append("non-finite data")

    if not settled:
        reasons.append("settling timeout")

    periods = []
    if np.any(config.linear_amplitude) and config.linear_frequency > 0.0:
        periods.append(1.0 / config.linear_frequency)
    if (np.any(config.rotational_amplitude)
            and config.rotational_frequency > 0.0):
        periods.append(1.0 / config.rotational_frequency)
    if periods and config.evaluation_seconds < max(periods):
        reasons.append("evaluation shorter than one disturbance period")

    accepted = bool(
        settled and metrics_computable and joint_limit_within_tolerance)
    return RunStatus(
        metrics_computable=metrics_computable,
        accepted=accepted,
        contact_observed=contact_observed,
        joint_limit_within_tolerance=joint_limit_within_tolerance,
        limit_penetration_rad=limit_penetration,
        warning_reasons=tuple(dict.fromkeys(reasons)),
    )
