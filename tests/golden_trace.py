"""Record and check a deterministic trace of the current controller.

This deliberately calls the current unrefactored modules directly. It is the
behavior lock for the migration, not the desired final control architecture.

Usage from the repository root:

    python -m tests.golden_trace --record
    python -m tests.golden_trace --check
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from controller import desired_pos, frames, servo
from controller.gain_sets import VERIFIED_BASELINE
from sim import motion, target_motion, targets, world


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "reactive_current.csv"
GOLDEN_MANIFEST = GOLDEN_DIR / "reactive_current.json"

BASELINE_REVISION = "d385173e150f438a80f1aac54871c47618b865a2"
TRACE_SCHEMA_VERSION = 1
STEPS = 250
RTOL = 1e-12
ATOL = 1e-12
ARMS = ("right", "left")

BASE_LINEAR_AMPLITUDE_M = np.array([0.18, 0.04, 0.05])
BASE_ROTATIONAL_AMPLITUDE_RAD = np.array([0.0, 0.0, -0.2])
BASE_LINEAR_FREQUENCY_HZ = 0.5
BASE_ROTATIONAL_FREQUENCY_HZ = 0.5

TARGET_LINEAR_AMPLITUDE_M = {
    "right": np.array([0.012, -0.008, 0.005]),
    "left": np.array([-0.009, 0.011, -0.004]),
}
TARGET_ROTATIONAL_AMPLITUDE_RAD = {
    "right": np.array([0.015, -0.010, 0.008]),
    "left": np.array([-0.012, 0.007, -0.009]),
}
TARGET_LINEAR_FREQUENCY_HZ = 0.7
TARGET_ROTATIONAL_FREQUENCY_HZ = 0.4


def _assert_baseline_gains():
    actual = (
        servo.KP_POS,
        servo.KP_ROT,
        servo.KD_POS,
        servo.KD_ROT,
        servo.K_NULL,
        servo.DAMPING,
    )
    expected = (
        VERIFIED_BASELINE.kp_pos,
        VERIFIED_BASELINE.kp_rot,
        VERIFIED_BASELINE.kd_pos,
        VERIFIED_BASELINE.kd_rot,
        VERIFIED_BASELINE.k_null,
        VERIFIED_BASELINE.damping,
    )
    if actual != expected:
        raise RuntimeError(
            f"golden trace requires baseline gains; got {actual}, expected {expected}"
        )


def _reset_current_code():
    _assert_baseline_gains()
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    target_motion.init_home()
    servo.init_ctrl()
    mujoco.mj_forward(world.model, world.data)


def _flatten(row, prefix, value):
    array = np.asarray(value)
    if array.ndim == 0:
        row[prefix] = array.item()
        return
    for index, item in enumerate(array.reshape(-1)):
        row[f"{prefix}_{index}"] = item.item()


def _cycle_rows(cycle, dt, base_twist, traces):
    torso_pos, torso_rot = frames.torso_pose()
    rows = []
    for side in ARMS:
        trace = traces[side]
        ee_pos, ee_rot = frames.ee_pose(side)
        ee_v, ee_w = frames.ee_velocity(side, base_twist, trace.J)
        row = {
            "cycle": cycle,
            "arm": side,
            "sample_time_s": world.data.time,
            "dt_s": dt,
        }
        _flatten(row, "torso_position_m", torso_pos)
        _flatten(row, "torso_rotation", torso_rot)
        _flatten(row, "torso_linear_velocity_m_s", base_twist[0])
        _flatten(row, "torso_angular_velocity_rad_s", base_twist[1])
        _flatten(row, "target_position_m", targets.target_position(side))
        _flatten(row, "target_quaternion_wxyz", targets.target_quat(side))
        _flatten(row, "ee_position_m", ee_pos)
        _flatten(row, "ee_rotation", ee_rot)
        _flatten(row, "ee_linear_velocity_m_s", ee_v)
        _flatten(row, "ee_angular_velocity_rad_s", ee_w)
        for name in trace.__dataclass_fields__:
            _flatten(row, name, getattr(trace, name))
        rows.append(row)
    return rows


def generate_rows():
    _reset_current_code()
    dt = float(world.model.opt.timestep)
    rows = []
    for cycle in range(STEPS):
        t = float(world.data.time)
        motion.set_torso_pose(
            t,
            BASE_LINEAR_AMPLITUDE_M,
            BASE_LINEAR_FREQUENCY_HZ,
            BASE_ROTATIONAL_AMPLITUDE_RAD,
            BASE_ROTATIONAL_FREQUENCY_HZ,
        )
        for side in ARMS:
            target_motion.set_target_pose(
                t,
                side,
                TARGET_LINEAR_AMPLITUDE_M[side],
                TARGET_LINEAR_FREQUENCY_HZ,
                TARGET_ROTATIONAL_AMPLITUDE_RAD[side],
                TARGET_ROTATIONAL_FREQUENCY_HZ,
            )
        mujoco.mj_kinematics(world.model, world.data)
        base_twist = motion.torso_twist_at(
            t,
            BASE_LINEAR_AMPLITUDE_M,
            BASE_LINEAR_FREQUENCY_HZ,
            BASE_ROTATIONAL_AMPLITUDE_RAD,
            BASE_ROTATIONAL_FREQUENCY_HZ,
        )
        traces = servo.apply_ctrl(dt, base_twist, ARMS)
        rows.extend(_cycle_rows(cycle, dt, base_twist, traces))
        mujoco.mj_step(world.model, world.data)
    return rows


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: (
                    value
                    if isinstance(value, str)
                    else format(float(value), ".17g")
                )
                for name, value in row.items()
            })


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record():
    rows = generate_rows()
    _write_csv(GOLDEN_CSV, rows)
    manifest = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "baseline_revision": BASELINE_REVISION,
        "csv": GOLDEN_CSV.name,
        "csv_sha256": _sha256(GOLDEN_CSV),
        "cycles": STEPS,
        "arms": list(ARMS),
        "rows": len(rows),
        "dt_s": float(world.model.opt.timestep),
        "duration_s": STEPS * float(world.model.opt.timestep),
        "comparison": {"relative_tolerance": RTOL, "absolute_tolerance": ATOL},
        "controller_output_field": "qdot_raw",
        "applied_command_field": "ctrl_after",
    }
    GOLDEN_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"recorded {len(rows)} rows to {GOLDEN_CSV}")
    print(f"sha256 {manifest['csv_sha256']}")


def _load_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _compare(expected_path, actual_path):
    expected = _load_csv(expected_path)
    actual = _load_csv(actual_path)
    if len(expected) != len(actual):
        raise AssertionError(
            f"row count differs: expected {len(expected)}, actual {len(actual)}"
        )
    if not expected:
        raise AssertionError("golden trace is empty")
    if tuple(expected[0]) != tuple(actual[0]):
        raise AssertionError("trace columns or column ordering differ")

    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        for name in expected_row:
            expected_value = expected_row[name]
            actual_value = actual_row[name]
            is_discrete = (
                name in ("cycle", "arm")
                or name.startswith("speed_saturated_")
                or name.startswith("lead_clamped_")
                or name.startswith("range_clamped_")
            )
            if is_discrete:
                if expected_value != actual_value:
                    raise AssertionError(
                        f"row {row_index} {name}: "
                        f"expected {expected_value}, actual {actual_value}"
                    )
                continue
            expected_number = float(expected_value)
            actual_number = float(actual_value)
            if not np.isclose(
                expected_number,
                actual_number,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=False,
            ):
                absolute_error = abs(actual_number - expected_number)
                relative_error = (
                    absolute_error / abs(expected_number)
                    if expected_number != 0.0
                    else float("inf")
                )
                raise AssertionError(
                    f"first divergence: row={row_index}, "
                    f"cycle={expected_row['cycle']}, arm={expected_row['arm']}, "
                    f"field={name}, expected={expected_number:.17g}, "
                    f"actual={actual_number:.17g}, "
                    f"absolute_error={absolute_error:.3e}, "
                    f"relative_error={relative_error:.3e}"
                )


def check():
    if not GOLDEN_CSV.is_file() or not GOLDEN_MANIFEST.is_file():
        raise FileNotFoundError("golden trace missing; run with --record first")
    manifest = json.loads(GOLDEN_MANIFEST.read_text())
    if _sha256(GOLDEN_CSV) != manifest["csv_sha256"]:
        raise AssertionError("golden CSV hash does not match its manifest")
    with tempfile.TemporaryDirectory(prefix="msc-golden-trace-") as directory:
        actual = Path(directory) / "actual.csv"
        _write_csv(actual, generate_rows())
        _compare(GOLDEN_CSV, actual)
    print(
        f"PASS: {manifest['rows']} rows match "
        f"(rtol={RTOL:g}, atol={ATOL:g})"
    )


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.record:
        record()
    else:
        check()


if __name__ == "__main__":
    main()
