"""Reproducibility identities and canonical-run gates."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from analysis import policy
from controller import servo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_SUFFIXES = {".py", ".xml"}
_SOURCE_ROOTS = ("analysis", "controller", "plotting", "sim", "tests")
_SOURCE_FILES = ("main.py", "requirements.txt", "AGENTS.md")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_named_files(paths, root=PROJECT_ROOT):
    digest = hashlib.sha256()
    for path in sorted((Path(value) for value in paths), key=lambda p: str(p)):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_paths(root=PROJECT_ROOT):
    root = Path(root)
    paths = []
    for directory in _SOURCE_ROOTS:
        base = root / directory
        if base.is_dir():
            paths.extend(
                path for path in base.rglob("*")
                if path.is_file() and path.suffix in _SOURCE_SUFFIXES
            )
    paths.extend(root / name for name in _SOURCE_FILES if (root / name).is_file())
    return tuple(sorted(set(paths)))


def source_sha256(root=PROJECT_ROOT):
    return _hash_named_files(source_paths(root), Path(root))


def analysis_sha256(root=PROJECT_ROOT):
    root = Path(root)
    paths = tuple((root / "analysis").glob("*.py"))
    return _hash_named_files(paths, root)


def _git(args, root=PROJECT_ROOT):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        text=True).stdout


def git_provenance(root=PROJECT_ROOT):
    root = Path(root)
    try:
        revision = _git(["rev-parse", "HEAD"], root).strip()
        status = _git(
            ["status", "--porcelain=v1", "--untracked-files=all"], root)
        branch = _git(["branch", "--show-current"], root).strip()
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": "unknown",
            "branch": "unknown",
            "worktree_clean": False,
            "worktree_status_sha256": "unknown",
        }
    return {
        "revision": revision,
        "branch": branch or "detached",
        "worktree_clean": not bool(status),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def dependency_versions():
    versions = {}
    for package in ("numpy", "matplotlib", "mujoco", "pin"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def environment_snapshot():
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "pinned_primary": dict(policy.PRIMARY_ENVIRONMENT),
    }


def primary_environment_matches(snapshot=None):
    snapshot = environment_snapshot() if snapshot is None else snapshot
    actual = {
        "python": snapshot["python"],
        "numpy": snapshot["dependencies"]["numpy"],
        "mujoco": snapshot["dependencies"]["mujoco"],
        "pinocchio": snapshot["dependencies"]["pin"],
        "matplotlib": snapshot["dependencies"]["matplotlib"],
    }
    return actual == policy.PRIMARY_ENVIRONMENT


def controller_configuration(gain_overrides=None):
    gains = {
        name: float(getattr(servo, name))
        for name in ("KP_POS", "KP_ROT", "KD_POS", "KD_ROT", "K_NULL",
                     "DAMPING")
    }
    if gain_overrides:
        gains.update({name: float(value)
                      for name, value in gain_overrides.items()})
    return {
        "gains": gains,
        "components": {
            "position_enabled": bool(servo.POSITION_ENABLED),
            "orientation_enabled": bool(servo.ORIENTATION_ENABLED),
            "velocity_enabled": bool(servo.VELOCITY_ENABLED),
        },
        "joint_speed_limits_rad_s": np.asarray(
            servo.QDOT_LIMIT, dtype=float).tolist(),
        "control_lead_limit_rad": float(servo.CTRL_LEAD),
        "command_interface": "integrated_clipped_joint_velocity_to_position",
    }


def experiment_identity(configuration, gain_overrides=None):
    value = {
        "evidence_schema_version": policy.EVIDENCE_SCHEMA_VERSION,
        "acceptance_policy_version": policy.ACCEPTANCE_POLICY_VERSION,
        "joint_limit_max_penetration_rad": (
            policy.JOINT_LIMIT_MAX_PENETRATION_RAD),
        "configuration": configuration,
        "controller": controller_configuration(gain_overrides),
        "source_sha256": source_sha256(),
        "analysis_sha256": analysis_sha256(),
        "scene_asset_sha256": _scene_asset_sha256(),
        "environment": environment_snapshot(),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return {**value, "identity_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _scene_asset_sha256(root=PROJECT_ROOT):
    root = Path(root)
    paths = [root / "sim" / "scene.xml"]
    paths.extend(sorted((root / "sim" / "assets").rglob("*")))
    return _hash_named_files([path for path in paths if path.is_file()], root)


def require_canonical_preconditions(log, git=None, environment=None):
    git = git_provenance() if git is None else git
    environment = environment_snapshot() if environment is None else environment
    if not log.accepted:
        raise RuntimeError("canonical runs require an accepted experiment log")
    if not git["worktree_clean"]:
        raise RuntimeError("canonical runs require a clean Git worktree")
    if not primary_environment_matches(environment):
        raise RuntimeError("canonical runs require the pinned primary environment")


def artifact_hashes(run_dir, final_outputs=()):
    run_dir = Path(run_dir)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    outputs = []
    for value in final_outputs:
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError(f"final output does not exist: {path}")
        outputs.append({"path": str(path), "sha256": sha256_file(path)})
    return artifacts, outputs
