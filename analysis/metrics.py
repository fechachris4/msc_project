"""Error statistics for the analysis scripts (millimetres).
"""

import numpy as np

from sim import world


def stats(log):
    """Full-run RMS, peak and rejection per arm, in millimetres."""
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
    """Rolling-window RMS, peak and rejection."""
    rms = float(np.sqrt(np.mean(e_norm_mm**2))) if e_norm_mm.size else 0.0
    peak = float(e_norm_mm.max()) if e_norm_mm.size else 0.0
    peak_base = float(base_norm_mm.max()) if base_norm_mm.size else 0.0
    rejection = ((1.0 - peak / peak_base) * 100.0
                 if peak_base > 0.0 else None)
    return rms, peak, rejection
