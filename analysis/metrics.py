"""One metric definition, shared by every full-run table and every live
readout in this repo: rejection = 1 - max_t|e(t)| / max_t|base_disp(t)|,
norm-based. stats()/print_stats() are the full-run table (originally
analysis/base_vs_error.py's); windowed_stats() is the same formula over
whatever slice of history a live caller (analysis/dashboard.py's rolling
window) passes in.

SI in, mm out: every function here takes metres and formats/returns mm.
"""

import numpy as np

from sim import world


def stats(log):
    """{side: {rms (3,) mm, peak (3,) mm, peak_norm mm, rejection %}} plus
    "peak_base" (mm, norm) — rejection = 1 - peak|e|/peak|base disp|,
    the norm-based thesis success metric. log: {"base_disp", "right_e",
    "left_e"} full-run arrays, metres, world frame."""
    base_mm = log["base_disp"] * 1000.0
    peak_base = float(np.linalg.norm(base_mm, axis=1).max())

    out = {"peak_base": peak_base}
    for side, key in (("right", "right_e"), ("left", "left_e")):
        e_mm = log[key] * 1000.0
        e_norm = np.linalg.norm(e_mm, axis=1)
        rms = np.sqrt(np.mean(e_mm**2, axis=0))
        peak = np.max(np.abs(e_mm), axis=0)
        peak_norm = float(e_norm.max())
        rejection = (1.0 - peak_norm / peak_base) * 100.0
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
        print(f"{side:8s}{cells}{d['peak_norm']:12.2f}"
              f"{d['rejection']:11.1f}%  [mm]")
    print(f"peak |base disp| = {st['peak_base']:.2f} mm")


def windowed_stats(e_norm_mm, base_norm_mm):
    """(rms, peak, rejection) over whatever window is passed in -- the
    caller decides how much history that is (e.g. analysis/dashboard.py's
    rolling ring buffer). Same rejection formula as stats(), just over a
    window instead of the full run."""
    rms = float(np.sqrt(np.mean(e_norm_mm**2))) if e_norm_mm.size else 0.0
    peak = float(e_norm_mm.max()) if e_norm_mm.size else 0.0
    peak_base = float(base_norm_mm.max()) if base_norm_mm.size else 0.0
    rejection = (1.0 - peak / peak_base) * 100.0 \
        if peak_base > 0 else float("nan")
    return rms, peak, rejection
