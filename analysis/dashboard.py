"""Live 7-panel control-loop dashboard: per-axis tracking, orientation,
saturation headroom, Jacobian conditioning, and joint/servo margins, for
one or both arms, with the shared live gain panel for tuning.

    python -m analysis.dashboard right
    python -m analysis.dashboard left
    python -m analysis.dashboard              # both arms, live (default)
    python -m analysis.dashboard both --save 10   # headless, 10 sim-seconds

Base motion follows the public research levers in sim.motion. Every
per-tick quantity is
recomputed from the same public functions apply_ctrl itself calls
(servo.pose_error, servo.twist_error, servo.qdot_from_error,
frames.jacobian_world) — the analysis/diagnose.py convention: the
dashboard shows exactly what the controller saw, nothing re-derived.

Panels (top to bottom), sharing the time axis:
  1-3. world x, y, z: signed per-axis error (mm, e = ref - actual) —
     grey signed base-displacement component + right + left EE error
     components, the same per-axis view as analysis/base_vs_error.py.
     Monospace readout on panel 1: RMS / peak / rejection % per side
     over the current window (rejection = 1 - peak|e|/peak|base disp|,
     both norm-based — analysis/metrics.py's windowed_stats).
  4. |e_rot| per side (deg) — orientation error, plotted nowhere else in
     this repo (only printed in diagnose.py's summary).
  5. Saturation headroom: 100 * max_i|qdot_raw_i|/QDOT_LIMIT_i per side,
     pre-clip, red 100% line (the one physical-limit line on this figure).
  6. sigma_min(J) per side, semilogy, grey lines at DAMPING=0.05 ("DLS
     active"), 0.01, 0.001 (diagnose.py's SV_THRESHOLDS).
  7. Margins (deg): min distance to jnt_range over the limited joints
     (solid) + max|ctrl-q| setpoint lead (dashed), per side; grey line at
     degrees(CTRL_LEAD) (a soft anti-windup bound, not hardware).

Right arm: solid, #D55E00. Left arm: dashed, #0072B2 (Okabe-Ito) — except
panel 7, where solid/dashed instead distinguishes margin from lead
(colour still carries which side).

The figure title and legend are static (no-jump rule): the suptitle
never rewrites itself mid-run, and y-axes only ever expand, never
shrink, so the picture doesn't visually jump while you watch it live.

Gain panel (live mode only): the shared plotting.gain_panel.GainPanel
(KP_POS, KP_ROT, KD_POS, KD_ROT, K_NULL, DAMPING) opens alongside the
dashboard. Every change flushes the ring buffers so a gain step doesn't
leave stale pre-change data in the rolling window.

A full-run log (every step, not just the rolling window) is kept
alongside the ring buffers; on exit (window closed) or after --save
completes, the analysis/metrics.py full-run RMS/peak/rejection table
prints on stdout, for both sides — pose_error is a pure readout, so it's
computed for both arms every step regardless of which are selected via
the CLI; only the selected arms are drawn on the live panels.
"""

import sys
from collections import deque
from pathlib import Path

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from analysis import metrics
from controller import desired_pos, frames, servo
from plotting.gain_panel import GainPanel
from plotting.style import C_BASE, SIDE_COLOR, SIDE_STYLE
from sim import motion, world

WINDOW_S = 15.0        # rolling window kept on screen, seconds
REDRAW_EVERY = 25      # sim steps between redraws (LivePlot's convention)
SV_THRESHOLDS = (0.05, 0.01, 0.001)  # diagnose.py's DLS-activity markers
SETTLE_SECONDS = 2.0   # --save only; base_vs_error.py's SETTLE pattern
AXIS_LABELS = ("x", "y", "z")

OUT = Path("analysis/output")


# --- pure per-tick helpers: importable and testable without a display ------


def headroom_frac(qdot_raw):
    """max_i |qdot_raw_i| / QDOT_LIMIT_i — the pre-clip demand as a
    fraction of each joint's speed limit (can exceed 1: qdot_raw is
    unclipped, this is what apply_ctrl is about to clip away)."""
    return float(np.max(np.abs(qdot_raw) / servo.QDOT_LIMIT))


def limit_margin_deg(q, q_low, q_high, limited):
    """Degrees from q to the nearer bound, minimized over the limited
    joints only (continuous joints have no range to be near — same
    convention as servo._centering / world.jnt_range)."""
    margin = np.minimum(q - q_low, q_high - q)
    return float(np.degrees(np.min(margin[limited])))


def ctrl_lead_deg(ctrl, q):
    """max_i |ctrl_i - q_i|, degrees — the anti-windup setpoint lead
    apply_ctrl clamps to CTRL_LEAD."""
    return float(np.degrees(np.max(np.abs(ctrl - q))))


def sigma_min(J):
    """Smallest singular value of the Jacobian — DLS is materially active
    once this falls near DAMPING (servo.qdot_from_error's lambda)."""
    return float(np.linalg.svd(J, compute_uv=False).min())


# --- the dashboard loop ------------------------------------------------


def run(arms, save_seconds=None):
    """Run the dashboard. arms: subset of world.SIDES to drive (via
    servo.apply_ctrl) and plot — an unselected arm holds posture and is
    not shown on the live panels (same selection semantics as main.py),
    though its pose_error is still logged for the final table. Both
    sides are always driven+plotted+logged when arms == world.SIDES
    (the default, "both"). save_seconds: if given, run headless for
    that many sim-seconds and save analysis/output/dashboard.png;
    otherwise run live with the gain panel until the figure window is
    closed."""
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep

    if save_seconds is not None:
        # Static settle before the scenario clock starts (base_vs_error.py's
        # SETTLE pattern): a fixed-length save has no ring buffer to age the
        # ~1.3 m boot transient out, so left in it would dominate peak error
        # and make rejection % meaningless. Live mode skips this — the ring
        # buffer ages the transient out on its own.
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(SETTLE_SECONDS / dt)):
            servo.apply_ctrl(dt, zero_twist)
            mujoco.mj_step(world.model, world.data)
    t_start = world.data.time  # phase 0 at motion start: no teleport

    maxlen = max(1, int(WINDOW_S / dt))
    n_steps = int(save_seconds / dt) if save_seconds is not None else None

    jlim = {s: world.jnt_range(s) for s in arms}

    # --- ring buffers (rolling window, live panels) -----------------------
    t_buf = deque(maxlen=maxlen)
    base_disp_buf = deque(maxlen=maxlen)      # (3,) mm, signed, per step
    e_pos_buf = {s: deque(maxlen=maxlen) for s in arms}  # (3,) mm, signed
    e_rot_buf = {s: deque(maxlen=maxlen) for s in arms}
    headroom_buf = {s: deque(maxlen=maxlen) for s in arms}
    sigma_buf = {s: deque(maxlen=maxlen) for s in arms}
    margin_buf = {s: deque(maxlen=maxlen) for s in arms}
    lead_buf = {s: deque(maxlen=maxlen) for s in arms}

    # --- full-run log (every step; analysis/metrics.py's shape) -----------
    full_log = {"t": [], "base_disp": [], "right_e": [], "left_e": []}

    # --- figure: 7 panels sharing the time axis ---------------------------
    if save_seconds is None:
        plt.ion()
    fig, axes = plt.subplots(
        7, 1, sharex=True, figsize=(9, 14),
        gridspec_kw={"height_ratios": [2, 2, 2, 2, 2, 2, 1]},
    )
    ax_x, ax_y, ax_z, ax_rot, ax_head, ax_sv, ax_marg = axes
    ax_axis = (ax_x, ax_y, ax_z)

    lines = {}
    for i, (ax, label) in enumerate(zip(ax_axis, AXIS_LABELS)):
        lines["base", i] = ax.plot([], [], color=C_BASE, linewidth=1.0,
                                   label="base disp" if i == 0 else None)[0]
        for s in arms:
            c, ls = SIDE_COLOR[s], SIDE_STYLE[s]
            lines["e", s, i] = ax.plot(
                [], [], color=c, linestyle=ls,
                label=f"{s} e_{label}" if i == 0 else None)[0]
        ax.axhline(0, color="0.85", linewidth=0.5, zorder=0)
        ax.set_ylabel(f"world {label} [mm]")

    readout = ax_x.text(0.99, 0.97, "", transform=ax_x.transAxes,
                        family="monospace", fontsize=8, va="top", ha="right")
    ax_x.legend(loc="upper left", fontsize=8)

    for s in arms:
        c, ls = SIDE_COLOR[s], SIDE_STYLE[s]
        lines["rot", s] = ax_rot.plot([], [], color=c, linestyle=ls,
                                      label=s)[0]
        lines["head", s] = ax_head.plot([], [], color=c, linestyle=ls,
                                        label=s)[0]
        lines["sv", s] = ax_sv.plot([], [], color=c, linestyle=ls,
                                    label=s)[0]
        # Panel 7 overrides the side->linestyle convention: solid/dashed
        # here distinguishes margin from lead, not side (colour still
        # does that) — see module docstring.
        lines["marg", s] = ax_marg.plot([], [], color=c, linestyle="-",
                                        label=f"{s} margin")[0]
        lines["lead", s] = ax_marg.plot([], [], color=c, linestyle="--",
                                        label=f"{s} lead")[0]

    ax_rot.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax_rot.set_ylabel("|e_rot| [deg]")

    ax_head.axhline(100.0, color="red", linewidth=0.8)
    ax_head.set_ylabel("headroom [%]")

    ax_sv.set_yscale("log")
    for th in SV_THRESHOLDS:
        ax_sv.axhline(th, color="0.6", linewidth=0.6)
    ax_sv.annotate("DLS active, λ", (0.01, SV_THRESHOLDS[0]),
                  xycoords=("axes fraction", "data"), fontsize=7,
                  color="0.4", va="bottom")
    ax_sv.set_ylabel("σ_min(J)")

    ax_marg.axhline(np.degrees(servo.CTRL_LEAD), color="0.6", linewidth=0.6)
    ax_marg.set_ylabel("margin [deg]")
    ax_marg.set_xlabel("sim time (s)")

    # Static suptitle (no-jump rule) — never rewritten mid-run.
    fig.suptitle("World-frame pose hold under base sway (e = ref − actual)")

    panel = None  # keep the GainPanel reference alive for the run's duration
    if save_seconds is None:
        def _flush():
            t_buf.clear()
            base_disp_buf.clear()
            for s in arms:
                e_pos_buf[s].clear()
                e_rot_buf[s].clear()
                headroom_buf[s].clear()
                sigma_buf[s].clear()
                margin_buf[s].clear()
                lead_buf[s].clear()
        panel = GainPanel(on_change=_flush)
        fig.tight_layout()
    else:
        fig.tight_layout()

    # Expand-only y-autoscale (no-jump rule): per-axes min/max only ever
    # grow. Same trade-off as plotting.live_plot.LivePlot: a boot
    # transient keeps a panel's scale enlarged for the rest of the run.
    ylim = {ax: [None, None] for ax in axes}

    def _apply_expand_only_ylim(ax):
        ax.relim()
        ax.autoscale_view(scaley=False)
        lo, hi = ax.dataLim.y0, ax.dataLim.y1
        cur_lo, cur_hi = ylim[ax]
        lo = lo if cur_lo is None else min(cur_lo, lo)
        hi = hi if cur_hi is None else max(cur_hi, hi)
        if lo < hi:
            ylim[ax] = [lo, hi]
            ax.set_ylim(lo, hi)

    def _redraw():
        t_arr = np.array(t_buf)
        base_arr = np.array(base_disp_buf) if base_disp_buf else \
            np.empty((0, 3))
        for i, ax in enumerate(ax_axis):
            lines["base", i].set_data(t_arr, base_arr[:, i]
                                      if base_arr.size else [])

        stats_lines = []
        for s in arms:
            e_arr = np.array(e_pos_buf[s]) if e_pos_buf[s] else \
                np.empty((0, 3))
            for i, ax in enumerate(ax_axis):
                lines["e", s, i].set_data(
                    t_arr, e_arr[:, i] if e_arr.size else [])
            lines["rot", s].set_data(t_arr, e_rot_buf[s])
            lines["head", s].set_data(t_arr, headroom_buf[s])
            lines["sv", s].set_data(t_arr, sigma_buf[s])
            lines["marg", s].set_data(t_arr, margin_buf[s])
            lines["lead", s].set_data(t_arr, lead_buf[s])

            e_norm = np.linalg.norm(e_arr, axis=1) if e_arr.size \
                else np.empty(0)
            base_norm = np.linalg.norm(base_arr, axis=1) if base_arr.size \
                else np.empty(0)
            rms, peak, rejection = metrics.windowed_stats(e_norm, base_norm)
            stats_lines.append(
                f"{s:>5s}  rms{rms:6.1f}  pk{peak:6.1f}  "
                f"rej{rejection:5.0f}%  [mm]"
            )
        readout.set_text("\n".join(stats_lines))

        for ax in axes:
            _apply_expand_only_ylim(ax)
        if save_seconds is None:
            plt.pause(0.001)  # also pumps the GainPanel's event loop

    # --- main loop (mirrors analysis/diagnose.py's ordering) -------------
    step = 0
    while True:
        if n_steps is not None:
            if step >= n_steps:
                break
        elif not plt.fignum_exists(fig.number):
            break

        t = world.data.time - t_start
        motion.set_torso_pose(t)
        # refresh xpos/xmat so every quantity below sees the torso pose
        # at t, not the previous step's (main.py / diagnose.py pattern)
        mujoco.mj_kinematics(world.model, world.data)
        # set_torso_pose (mocap write) and torso_twist_at (feedforward)
        # must stay a matched pair — same scenario, same instant t.
        base_twist = motion.torso_twist_at(t)

        base_pos, _ = frames.torso_pose()
        base_disp = base_pos - motion.HOME_POS  # (3,) m, signed

        # Pre-control state: exactly what apply_ctrl is about to use.
        # damping=servo.DAMPING is passed explicitly (unlike diagnose.py,
        # which pins its gains) so the DAMPING slider is reflected here
        # too — qdot_from_error's own damping default binds at def time,
        # same reason apply_ctrl now passes it explicitly (servo.py fix).
        tick = {}
        for s in arms:
            J = frames.jacobian_world(s)
            e_pos, e_rot = servo.pose_error(s)
            e_v, e_w = servo.twist_error(s, base_twist, J)
            q = world.data.qpos[frames.qpos_adrs[s]].copy()
            qdot_raw = servo.qdot_from_error(
                J, e_pos, e_rot, e_v, e_w, q,
                servo._Q_MID[s], servo._K_NULL_VEC[s],
                damping=servo.DAMPING)
            tick[s] = dict(J=J, e_pos=e_pos, e_rot=e_rot, q=q,
                           qdot_raw=qdot_raw)

        servo.apply_ctrl(dt, base_twist, arms)

        # Post-control: the setpoints the servos will now chase.
        for s in arms:
            ctrl = world.data.ctrl[world.ctrl_adrs[s]]
            tick[s]["lead_deg"] = ctrl_lead_deg(ctrl, tick[s]["q"])

        mujoco.mj_step(world.model, world.data)

        # Full-run log: both sides always, for the final metrics table
        # (analysis.metrics expects both "right_e"/"left_e" — pose_error
        # is a pure readout, cheap to compute even for an arm that isn't
        # driven/shown this run).
        full_log["t"].append(t)
        full_log["base_disp"].append(base_disp.copy())
        for side in world.SIDES:
            if side in tick:
                e_pos_full = tick[side]["e_pos"]
            else:
                e_pos_full, _ = servo.pose_error(side)
            full_log[f"{side}_e"].append(e_pos_full.copy())

        t_buf.append(t)
        base_disp_buf.append(base_disp * 1000.0)
        for s in arms:
            low, high, limited = jlim[s]
            e_pos_buf[s].append(tick[s]["e_pos"] * 1000.0)
            e_rot_buf[s].append(np.degrees(np.linalg.norm(tick[s]["e_rot"])))
            headroom_buf[s].append(100.0 * headroom_frac(tick[s]["qdot_raw"]))
            sigma_buf[s].append(sigma_min(tick[s]["J"]))
            margin_buf[s].append(
                limit_margin_deg(tick[s]["q"], low, high, limited))
            lead_buf[s].append(tick[s]["lead_deg"])

        step += 1
        if n_steps is None and step % REDRAW_EVERY == 0:
            _redraw()

    _redraw()

    if len(full_log["t"]) > 0:
        full_log = {k: np.asarray(v) for k, v in full_log.items()}
        st = metrics.stats(full_log)
        print(f"\nFull run: {full_log['t'][-1]:.1f} s "
              f"({len(full_log['t'])} samples)")
        metrics.print_stats(st)

    if n_steps is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "dashboard.png"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    return fig


if __name__ == "__main__":
    _args = sys.argv[1:]
    _save_seconds = None
    if "--save" in _args:
        _i = _args.index("--save")
        _save_seconds = float(_args[_i + 1])
        del _args[_i:_i + 2]
    _side_arg = _args[0] if _args else "both"
    assert _side_arg in ("right", "left", "both"), \
        "usage: python -m analysis.dashboard [right|left|both] [--save T]"
    _arms = world.SIDES if _side_arg == "both" else (_side_arg,)

    run(_arms, save_seconds=_save_seconds)
