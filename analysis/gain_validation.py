"""Fail-closed joint gain validation and explicit promotion boundary."""

from dataclasses import dataclass
import json
from pathlib import Path

from controller.gain_sets import EXPLORATORY_CANDIDATES


POLICY_PATH = Path(__file__).with_name("joint_gain_validation_policy.json")


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]


def load_policy(path=POLICY_PATH):
    policy = json.loads(Path(path).read_text())
    required = {
        "schema_version", "status", "scenario_matrix",
        "acceptance_thresholds", "required_dimensions",
        "diagnostic_evidence_kinds", "promotion_enabled", "reason",
    }
    if set(policy) != required:
        raise ValueError("joint validation policy has an unexpected schema")
    if not isinstance(policy["scenario_matrix"], list):
        raise ValueError("scenario_matrix must be a list")
    if not isinstance(policy["acceptance_thresholds"], dict):
        raise ValueError("acceptance_thresholds must be an object")
    return policy


def exploratory_report(candidate_name, evidence=()):
    if candidate_name not in EXPLORATORY_CANDIDATES:
        raise ValueError(f"unknown exploratory candidate: {candidate_name!r}")
    return {
        "schema_version": load_policy()["schema_version"],
        "classification": "exploratory",
        "candidate_name": candidate_name,
        "gains": EXPLORATORY_CANDIDATES[candidate_name].as_overrides(),
        "evidence": list(evidence),
        "promotion_requested": False,
    }


def promotion_decision(report, policy=None):
    policy = load_policy() if policy is None else policy
    reasons = []
    if policy["status"] != "approved":
        reasons.append("joint validation policy is pending")
    if not policy["scenario_matrix"]:
        reasons.append("joint validation scenario matrix is empty")
    if not policy["acceptance_thresholds"]:
        reasons.append("joint validation thresholds are empty")
    if not policy["promotion_enabled"]:
        reasons.append("gain promotion is disabled")
    if report.get("classification") != "joint_validation":
        reasons.append("evidence is exploratory, not joint validation")
    evidence_kinds = {
        item.get("kind") for item in report.get("evidence", [])
        if isinstance(item, dict)
    }
    diagnostic = evidence_kinds.intersection(
        policy["diagnostic_evidence_kinds"])
    if diagnostic and evidence_kinds <= set(policy["diagnostic_evidence_kinds"]):
        reasons.append("diagnostic sweeps cannot promote gains")
    if not report.get("all_thresholds_passed", False):
        reasons.append("approved thresholds have not all passed")
    if not report.get("canonical_manifest_sha256"):
        reasons.append("canonical joint-validation evidence is missing")
    return PromotionDecision(not reasons, tuple(reasons))


def require_promotion_eligible(report, policy=None):
    decision = promotion_decision(report, policy)
    if not decision.eligible:
        raise RuntimeError("gain promotion refused: " + "; ".join(decision.reasons))
    return decision
