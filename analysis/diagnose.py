"""Diagnose the right-arm tracking failure under torso roll. Read-only:
the controller is never modified — every logged quantity is recomputed
from sim state with the same public functions apply_ctrl itself calls
(servo.pose_error, frames.jacobian_world, servo.qdot_from_error), so the
log shows exactly what the controller saw each cycle.

    python -m analysis.diagnose

Outputs: analysis/output/diagnosis.npz, 8 figures (PNG), and an event
table on stdout. No fixes, no controller changes — evidence only.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller import desired_pos, frames, servo
from sim import motion, targets, world

# Pinned failure scenario (the levers observed to break the right arm);
# pinned here so the diagnosis reproduces even if sim/motion.py changes.
LIN_A = np.array([0.1, 0.0, 0.0])   # m
ROT_A = np.array([0.4, 0.0, 0.0])   # rad (23 deg roll)
FREQ = 0.1                          # Hz, both
SIM_SECONDS = 40.0                  # 4 periods

SV_THRESHOLDS = (0.05, 0.01, 0.001)
OUT = Path("analysis/output")

# Okabe-Ito
C_RIGHT = "#D55E00"
C_LEFT = "#0072B2"
C_XYZ = ("#D55E00", "#009E73", "#0072B2")


def _dof_adrs(side):
    adrs = []
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        adrs.append(int(world.model.jnt_dofadr[jnt_id]))
    return adrs


def _jnt_range(side):
    """(low, high) per joint; unlimited (continuous) joints get ±inf."""
    low = np.full(7, -np.inf)
    high = np.full(7, np.inf)
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        if world.model.jnt_limited[jnt_id]:
            low[i - 1], high[i - 1] = world.model.jnt_range[jnt_id]
    return low, high


def _arm_body_ids(side):
    ids = set()
    for body_id in range(world.model.nbody):
        name = mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY,
                                 body_id)
        if name and name.startswith(f"{side}_"):
            ids.add(body_id)
    return ids


def _contacts_touching(body_ids):
    n = 0
    for c in world.data.contact[: world.data.ncon]:
        b1 = world.model.geom_bodyid[c.geom1]
        b2 = world.model.geom_bodyid[c.geom2]
        if b1 in body_ids or b2 in body_ids:
            n += 1
    return n


def run():
    """Run the pinned scenario headless; return {side: {key: array}}."""
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep
    n_steps = int(SIM_SECONDS / dt)
    dofs = {s: _dof_adrs(s) for s in world.SIDES}
    jlims = {s: _jnt_range(s) for s in world.SIDES}
    bodies = {s: _arm_body_ids(s) for s in world.SIDES}

    log = {s: {
        "t": np.empty(n_steps),
        "target_pos": np.empty((n_steps, 3)),
        "ee_pos": np.empty((n_steps, 3)),
        "e_pos": np.empty((n_steps, 3)),
        "e_norm": np.empty(n_steps),
        "e_rot": np.empty((n_steps, 3)),
        "e_rot_norm": np.empty(n_steps),
        "sv": np.empty((n_steps, 6)),
        "min_sv": np.empty(n_steps),
        "cond": np.empty(n_steps),
        "rank": np.empty(n_steps, dtype=int),
        "q": np.empty((n_steps, 7)),
        "qdot_meas": np.empty((n_steps, 7)),
        "qdot_raw": np.empty((n_steps, 7)),
        "qdot_sat": np.empty((n_steps, 7)),
        "jlim_dist": np.empty((n_steps, 7)),
        "ctrl_dist": np.empty((n_steps, 7)),
        "servo_lag": np.empty((n_steps, 7)),
        "v_des": np.empty((n_steps, 6)),
        "v_ach": np.empty((n_steps, 6)),
        "contacts": np.empty(n_steps, dtype=int),
    } for s in world.SIDES}

    for k in range(n_steps):
        t = world.data.time
        motion.set_torso_pose(t, linear_amplitude=LIN_A,
                              linear_frequency=FREQ,
                              rotational_amplitude=ROT_A,
                              rotational_frequency=FREQ)

        # Pre-control state: exactly what apply_ctrl is about to use.
        for s in world.SIDES:
            L = log[s]
            e_pos, e_rot = servo.pose_error(s)
            J = frames.jacobian_world(s)
            sv = np.linalg.svd(J, compute_uv=False)
            qdot_raw = servo.qdot_from_error(J, e_pos, e_rot)
            q = world.data.qpos[frames.qpos_adrs[s]].copy()
            qdot_meas = world.data.qvel[dofs[s]].copy()
            low, high = jlims[s]

            L["t"][k] = t
            L["target_pos"][k] = targets.target_position(s)
            L["ee_pos"][k] = frames.ee_pose(s)[0]
            L["e_pos"][k] = e_pos
            L["e_norm"][k] = np.linalg.norm(e_pos)
            L["e_rot"][k] = e_rot
            L["e_rot_norm"][k] = np.linalg.norm(e_rot)
            L["sv"][k] = sv
            L["min_sv"][k] = sv.min()
            L["cond"][k] = sv.max() / max(sv.min(), 1e-300)
            L["rank"][k] = int(np.sum(sv > 1e-10))
            L["q"][k] = q
            L["qdot_meas"][k] = qdot_meas
            L["qdot_raw"][k] = qdot_raw
            L["qdot_sat"][k] = np.clip(qdot_raw, -servo.QDOT_LIMIT,
                                       servo.QDOT_LIMIT)
            L["jlim_dist"][k] = np.minimum(q - low, high - q)
            L["v_des"][k] = np.concatenate([servo.KP_POS * e_pos,
                                            servo.KP_ROT * e_rot])
            L["v_ach"][k] = J @ qdot_meas

        servo.apply_ctrl(dt)

        # Post-control: the setpoints the servos will now chase.
        for s in world.SIDES:
            ctrl = world.data.ctrl[world.ctrl_adrs[s]].copy()
            lo, hi = servo._BOUNDS[s]
            log[s]["ctrl_dist"][k] = np.minimum(ctrl - lo, hi - ctrl)
            log[s]["servo_lag"][k] = ctrl - log[s]["q"][k]

        mujoco.mj_step(world.model, world.data)

        for s in world.SIDES:
            log[s]["contacts"][k] = _contacts_touching(bodies[s])

    return log


def sv_events(log):
    """{side: {threshold: first crossing time or None}}."""
    ev = {}
    for s in world.SIDES:
        ev[s] = {}
        for th in SV_THRESHOLDS:
            below = np.nonzero(log[s]["min_sv"] < th)[0]
            ev[s][th] = log[s]["t"][below[0]] if below.size else None
    return ev


SETTLE = 3.0  # s — the startup transient (singular zero config, 1.3 m
              # from target) is excluded from every figure


def _shade_contacts(ax, t, contacts):
    """Light red bands over the intervals where the arm is in contact."""
    on = contacts > 0
    starts, stops = [], []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            starts.append(t[i])
            stops.append(t[j - 1])
            i = j
        else:
            i += 1
    for a, b in zip(starts, stops):
        ax.axvspan(a, b, color="red", alpha=0.12, linewidth=0)


def _spearman(x, y):
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)
    rx, ry = rank(x), rank(y)
    return np.corrcoef(rx, ry)[0, 1]


def make_figures(log, ev):
    OUT.mkdir(parents=True, exist_ok=True)
    r, l = log["right"], log["left"]
    t = r["t"]
    m = t >= SETTLE
    tm = t[m]
    mean_r = r["e_norm"][m].mean() * 1000
    mean_l = l["e_norm"][m].mean() * 1000
    in_contact = r["contacts"][m] > 0

    # A — the failure story: roll (cause), error (effect), servo lag
    # (mechanism), sigma_min (ruled-out alternative), one time axis.
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10),
                             layout="constrained")
    roll = np.degrees(motion.sine_offset(tm, ROT_A[0], FREQ))
    axes[0].plot(tm, roll, color="black", linewidth=1.2)
    axes[0].axhline(0, color="0.7", linewidth=0.5, zorder=0)
    axes[0].set_ylabel("torso roll [deg]")

    axes[1].plot(tm, r["e_norm"][m] * 1000, color=C_RIGHT, label="right")
    axes[1].plot(tm, l["e_norm"][m] * 1000, color=C_LEFT, linestyle="--",
                 label="left (control)")
    axes[1].set_ylabel("|e_pos| [mm]")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(tm, np.abs(r["servo_lag"][m]).max(axis=1), color=C_RIGHT,
                 label="right")
    axes[2].plot(tm, np.abs(l["servo_lag"][m]).max(axis=1), color=C_LEFT,
                 linestyle="--", label="left (control)")
    axes[2].set_ylabel("servo lag\nmax|ctrl-q| [rad]")

    axes[3].semilogy(tm, r["min_sv"][m], color=C_RIGHT, label="right")
    axes[3].semilogy(tm, l["min_sv"][m], color=C_LEFT, linestyle="--",
                     label="left (control)")
    for th in SV_THRESHOLDS:
        axes[3].axhline(th, color="red", linewidth=0.6, alpha=0.5)
    axes[3].set_ylim(1e-5, 1)
    axes[3].set_ylabel("sigma_min(J)")
    axes[3].set_xlabel("time [s]")

    for ax in axes:
        _shade_contacts(ax, tm, r["contacts"][m])
    frac = np.mean(in_contact)
    axes[0].set_title(
        f"Right-arm failure follows torso contact (red bands, "
        f"{frac:.0%} of t>{SETTLE:.0f}s):\nmean |e| right {mean_r:.0f} mm "
        f"vs left {mean_l:.0f} mm; sigma_min near-identical between arms")
    fig.savefig(OUT / "A_failure_story.png", dpi=200)

    # B — is conditioning correlated with error, once contact is separated?
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    classes = (
        ("right, in contact", r["min_sv"][m][in_contact],
         r["e_norm"][m][in_contact] * 1000, "red", 0.5),
        ("right, contact-free", r["min_sv"][m][~in_contact],
         r["e_norm"][m][~in_contact] * 1000, "0.55", 0.4),
        ("left (control)", l["min_sv"][m], l["e_norm"][m] * 1000,
         C_LEFT, 0.3),
    )
    for name, x, y, color, alpha in classes:
        rho = _spearman(x, y) if len(x) > 1 else np.nan
        ax.scatter(x, y, s=3, color=color, alpha=alpha,
                   label=f"{name} (Spearman rho={rho:+.2f}, n={len(x)})")
    ax.set_xlabel("sigma_min(J)")
    ax.set_ylabel("|e_pos| [mm]")
    ax.set_title("Large errors occur in contact, not at low sigma_min --\n"
                 "conditioning does not explain the failure")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "B_conditioning_vs_error.png", dpi=200)

    # C — joint limit margin: normalized position inside jnt_range.
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    joint_cols = {1: "#D55E00", 3: "#009E73", 5: "#0072B2"}  # q2, q4, q6
    for s, style in (("right", "-"), ("left", "--")):
        low, high = _jnt_range(s)
        for j, col in joint_cols.items():
            frac_range = (log[s]["q"][m, j] - low[j]) / (high[j] - low[j])
            ax.plot(tm, frac_range, style, color=col, linewidth=1.0,
                    label=f"q{j + 1} {s}")
    for y in (0.0, 1.0):
        ax.axhline(y, color="red", linewidth=1.0)
    _shade_contacts(ax, tm, r["contacts"][m])
    low_r, high_r = _jnt_range("right")
    q6r = (log["right"]["q"][m, 5] - low_r[5]) / (high_r[5] - low_r[5])
    pin = np.mean(q6r < 0.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position in jnt_range [0=low, 1=high]")
    ax.set_title(f"Limited joints only: right q6 pinned at its lower limit "
                 f"{pin:.0%} of t>{SETTLE:.0f}s")
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(OUT / "C_joint_limit_margin.png", dpi=200)

    # D — is velocity saturation active, and when?
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for s, col, style in (("right", C_RIGHT, "-"), ("left", C_LEFT, "--")):
        usage = np.max(np.abs(log[s]["qdot_raw"][m]) / servo.QDOT_LIMIT,
                       axis=1)
        ax.plot(tm, usage, style, color=col, linewidth=1.0, label=s)
    ax.axhline(1.0, color="red", linewidth=1.0)
    _shade_contacts(ax, tm, r["contacts"][m])
    sat_r = np.mean(np.any(np.abs(r["qdot_raw"][m]) > servo.QDOT_LIMIT,
                           axis=1))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("max_j |qdot_raw_j| / limit_j")
    ax.set_title(f"Commanded velocity vs limit: right saturates {sat_r:.0%} "
                 f"of t>{SETTLE:.0f}s, inside the contact windows")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "D_qdot_saturation.png", dpi=200)

    # E — appendix: per-joint positions with limits (cropped).
    for s, cs in (("right", C_RIGHT), ("left", C_LEFT)):
        L = log[s]
        low, high = _jnt_range(s)
        fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 12),
                                 layout="constrained")
        for j in range(7):
            ax = axes[j]
            ax.plot(tm, L["q"][m, j], color=cs, linewidth=1.0)
            for lim in (low[j], high[j]):
                if np.isfinite(lim):
                    ax.axhline(lim, color="red", linewidth=0.8)
            ax.set_ylabel(f"q{j + 1} [rad]")
        axes[-1].set_xlabel("time [s]")
        near = [int(j) + 1 for j in
                np.nonzero(np.min(L["jlim_dist"][m], axis=0) < 0.05)[0]]
        fig.suptitle(f"{s} arm joint positions (red: jnt_range); "
                     f"within 0.05 rad of a limit: {near or 'none'}")
        fig.savefig(OUT / f"E_appendix_joints_{s}.png", dpi=200)

    # F — appendix: right per-joint qdot before/after saturation (cropped).
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 12),
                             layout="constrained")
    for j in range(7):
        ax = axes[j]
        ax.plot(tm, r["qdot_raw"][m, j], color="0.6", linewidth=0.8,
                label="commanded (raw)" if j == 0 else None)
        ax.plot(tm, r["qdot_sat"][m, j], color=C_RIGHT, linewidth=1.0,
                label="after saturation" if j == 0 else None)
        ax.axhline(servo.QDOT_LIMIT[j], color="red", linewidth=0.8)
        ax.axhline(-servo.QDOT_LIMIT[j], color="red", linewidth=0.8)
        ax.set_ylabel(f"qdot{j + 1} [rad/s]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Right arm commanded qdot per joint (red: Gen3 speed "
                 "limits)")
    fig.savefig(OUT / "F_appendix_qdot_right.png", dpi=200)

    # G — appendix: y-z trajectory (the roll plane), zoomed on the target.
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    sc = ax.scatter(r["ee_pos"][m, 1], r["ee_pos"][m, 2], c=tm, s=2,
                    cmap="viridis")
    ax.scatter(r["ee_pos"][m, 1][in_contact], r["ee_pos"][m, 2][in_contact],
               s=4, color="red", alpha=0.5, label="in contact")
    ax.plot(r["target_pos"][0, 1], r["target_pos"][0, 2], "x",
            color="black", markersize=12, markeredgewidth=2.5,
            label="target")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Right EE in the roll plane (y-z), t>=3s: contact samples "
                 "(red)\nsit on the far side of the excursion")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="time [s]")
    fig.savefig(OUT / "G_appendix_ee_yz.png", dpi=200)

    plt.close("all")


def print_summary(log, ev):
    for s in world.SIDES:
        L = log[s]
        print(f"\n=== {s} arm ===")
        print(f"  |e_pos|  peak {L['e_norm'].max() * 1000:8.1f} mm   "
              f"final {L['e_norm'][-1] * 1000:8.1f} mm")
        print(f"  |e_rot|  peak {np.degrees(L['e_rot_norm'].max()):8.1f} deg  "
              f"final {np.degrees(L['e_rot_norm'][-1]):8.1f} deg")
        print(f"  sigma_min floor {L['min_sv'].min():.3e}   "
              f"cond peak {L['cond'].max():.3e}   "
              f"rank min {L['rank'].min()}")
        print(f"  min dist to jnt_range  {np.nanmin(L['jlim_dist']):.4f} rad "
              f"(joint {int(np.unravel_index(np.nanargmin(L['jlim_dist']), L['jlim_dist'].shape)[1]) + 1})")
        print(f"  min dist to ctrlrange  {np.min(L['ctrl_dist']):.4f} rad "
              f"(joint {int(np.unravel_index(np.argmin(L['ctrl_dist']), L['ctrl_dist'].shape)[1]) + 1})")
        print(f"  qdot saturation: any-joint fraction "
              f"{np.mean(np.any(np.abs(L['qdot_raw']) > servo.QDOT_LIMIT, axis=1)):.2%}")
        print(f"  servo lag |ctrl-q| peak {np.abs(L['servo_lag']).max():.3f} rad")
        print(f"  contacts: max {L['contacts'].max()} "
              f"(steps with any: {np.mean(L['contacts'] > 0):.2%})")
        print(f"  sigma_min crossings: " + ", ".join(
            f"<{th:g}: {f'{ev[s][th]:.1f}s' if ev[s][th] is not None else '—'}"
            for th in SV_THRESHOLDS))


def main():
    log = run()
    ev = sv_events(log)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "diagnosis.npz",
        **{f"{s}_{k}": v for s in world.SIDES for k, v in log[s].items()},
    )
    make_figures(log, ev)
    print_summary(log, ev)
    print(f"\nSaved {OUT}/diagnosis.npz and figures A–G.")


if __name__ == "__main__":
    main()
