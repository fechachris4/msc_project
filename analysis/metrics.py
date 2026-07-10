"""Evaluation-only experiment metrics and legacy analysis wrappers.

Experiment metrics stay in SI units and include units in every numeric key.
The legacy helpers retain their millimetre-based return contract.
"""

import numpy as np

from sim import world


_AXES = ("x", "y", "z")


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_vector(values):
    return [_finite_float(value) for value in np.asarray(values).reshape(-1)]


def _empty_arm_metrics():
    return {
        "position_error_axis_mean_m": None,
        "position_error_norm_mean_m": None,
        "position_error_axis_rmse_m": None,
        "position_error_norm_rmse_m": None,
        "position_error_axis_abs_peak_m": None,
        "position_error_norm_peak_m": None,
        "rotation_error_norm_mean_rad": None,
        "rotation_error_norm_rmse_rad": None,
        "rotation_error_norm_peak_rad": None,
        "rejection_pct": None,
        "velocity_saturation_overall_pct": None,
        "joint_margin_min_rad": None,
        "joint_margin_p01_rad": None,
        "contact_occupancy_total_pct": None,
        "torso_contact_occupancy_pct": None,
    }


def _side_contact_mask(contact_pairs, side):
    prefix = f"{side}_"
    return np.asarray([
        any(
            any(endpoint.split("/", 1)[0].startswith(prefix)
                for endpoint in pair.split(" <> "))
            for pair in pairs
        )
        for pairs in contact_pairs
    ], dtype=bool)


def _arm_metrics(log, side, mask, peak_base):
    if not np.any(mask):
        return _empty_arm_metrics()

    e_pos = np.asarray(log.arm_data[side]["e_pos"], dtype=float)[mask]
    e_norm = np.linalg.norm(e_pos, axis=1)
    axis_mean = np.mean(e_pos, axis=0)
    axis_rmse = np.sqrt(np.mean(e_pos**2, axis=0))
    axis_peak = np.max(np.abs(e_pos), axis=0)
    peak_norm = _finite_float(np.max(e_norm))

    e_rot = np.asarray(log.arm_data[side]["e_rot"], dtype=float)[mask]
    rot_norm = np.linalg.norm(e_rot, axis=1)
    if peak_base is None or peak_base <= 0.0 or peak_norm is None:
        rejection = None
    else:
        rejection = _finite_float((1.0 - peak_norm / peak_base) * 100.0)

    saturated = np.asarray(
        log.arm_data[side]["speed_saturated"], dtype=bool)[mask]
    margins = np.asarray(log.arm_data[side]["joint_margin"], dtype=float)[mask]
    finite_margins = margins[np.isfinite(margins)]
    if finite_margins.size:
        margin_min = _finite_float(np.min(finite_margins))
        margin_p01 = _finite_float(np.percentile(finite_margins, 1.0))
    else:
        margin_min = None
        margin_p01 = None

    torso_contact = np.asarray(log.torso_contact[side], dtype=bool)[mask]
    side_contacts = _side_contact_mask(log.contact_pairs, side)[mask]
    return {
        "position_error_axis_mean_m": _finite_vector(axis_mean),
        "position_error_norm_mean_m": _finite_float(np.mean(e_norm)),
        "position_error_axis_rmse_m": _finite_vector(axis_rmse),
        "position_error_norm_rmse_m": _finite_float(
            np.sqrt(np.mean(e_norm**2))),
        "position_error_axis_abs_peak_m": _finite_vector(axis_peak),
        "position_error_norm_peak_m": peak_norm,
        "rotation_error_norm_mean_rad": _finite_float(np.mean(rot_norm)),
        "rotation_error_norm_rmse_rad": _finite_float(
            np.sqrt(np.mean(rot_norm**2))),
        "rotation_error_norm_peak_rad": _finite_float(np.max(rot_norm)),
        "rejection_pct": rejection,
        "velocity_saturation_overall_pct": _finite_float(
            np.mean(saturated) * 100.0) if saturated.size else None,
        "joint_margin_min_rad": margin_min,
        "joint_margin_p01_rad": margin_p01,
        "contact_occupancy_total_pct": _finite_float(
            np.mean(side_contacts) * 100.0),
        "torso_contact_occupancy_pct": _finite_float(
            np.mean(torso_contact) * 100.0) if torso_contact.size else None,
    }


def _aggregate_metrics(log, mask):
    sample_count = int(np.count_nonzero(mask))
    if sample_count:
        base = np.asarray(log.base_displacement, dtype=float)[mask]
        peak_base = _finite_float(np.max(np.linalg.norm(base, axis=1)))
        contacts = np.asarray(log.contact_count)[mask]
        contact_occupancy = _finite_float(np.mean(contacts > 0) * 100.0)
    else:
        peak_base = None
        contact_occupancy = None
    return {
        "evaluation_sample_count": sample_count,
        "peak_base_displacement_m": peak_base,
        "contact_occupancy_total_pct": contact_occupancy,
        "arms": {
            side: _arm_metrics(log, side, mask, peak_base)
            for side in log.arms
        },
    }


def experiment_metrics(log):
    """Return JSON-safe evaluation metrics for a persisted experiment log."""
    evaluation_mask = np.asarray(log.evaluation_mask, dtype=bool)
    result = _aggregate_metrics(log, evaluation_mask)
    result["gain_segments"] = {
        str(int(segment)): _aggregate_metrics(
            log, evaluation_mask & (np.asarray(log.gain_segment) == segment))
        for segment in np.unique(np.asarray(log.gain_segment)[evaluation_mask])
    }
    return result


def flatten_metrics(values):
    """Flatten nested metric dictionaries for a single-row CSV artifact."""
    flattened = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            labels = _AXES if len(value) == len(_AXES) else range(len(value))
            for label, child in zip(labels, value):
                visit(f"{prefix}.{label}", child)
        else:
            flattened[prefix] = value

    visit("", values)
    return flattened


def stats(log):
    """Return the legacy full-run millimetre metric dictionary."""
    base_mm = np.asarray(log["base_disp"], dtype=float) * 1000.0
    peak_base = (float(np.linalg.norm(base_mm, axis=1).max())
                 if len(base_mm) else 0.0)

    out = {"peak_base": peak_base}
    for side, key in (("right", "right_e"), ("left", "left_e")):
        e_mm = np.asarray(log[key], dtype=float) * 1000.0
        if len(e_mm):
            e_norm = np.linalg.norm(e_mm, axis=1)
            rms = np.sqrt(np.mean(e_mm**2, axis=0))
            peak = np.max(np.abs(e_mm), axis=0)
            peak_norm = float(e_norm.max())
        else:
            rms = np.zeros(3)
            peak = np.zeros(3)
            peak_norm = 0.0
        rejection = ((1.0 - peak_norm / peak_base) * 100.0
                     if peak_base > 0.0 else None)
        out[side] = dict(rms=rms, peak=peak, peak_norm=peak_norm,
                         rejection=rejection)
    return out


def print_stats(st):
    print(f"{'':8s}{'RMS x':>10s}{'y':>10s}{'z':>10s}"
          f"{'peak x':>10s}{'y':>10s}{'z':>10s}"
          f"{'|e| peak':>12s}{'rejection':>12s}")
    for side in world.SIDES:
        d = st[side]
        cells = "".join(f"{v:10.2f}" for v in (*d["rms"], *d["peak"]))
        rejection = (f"{d['rejection']:11.1f}%"
                     if d["rejection"] is not None else f"{'n/a':>12s}")
        print(f"{side:8s}{cells}{d['peak_norm']:12.2f}{rejection}  [mm]")
    print(f"peak |base disp| = {st['peak_base']:.2f} mm")


def windowed_stats(e_norm_mm, base_norm_mm):
    """Return legacy rolling-window RMS, peak, and rejection metrics."""
    rms = float(np.sqrt(np.mean(e_norm_mm**2))) if e_norm_mm.size else 0.0
    peak = float(e_norm_mm.max()) if e_norm_mm.size else 0.0
    peak_base = float(base_norm_mm.max()) if base_norm_mm.size else 0.0
    rejection = ((1.0 - peak / peak_base) * 100.0
                 if peak_base > 0.0 else None)
    return rms, peak, rejection
