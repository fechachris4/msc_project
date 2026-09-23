"""The two hardware figures in the README (media/hw_walking_trial.png,
media/hw_rejection.png), drawn from the figure-data CSVs of my thesis
analysis of the treadmill trials. Participant data are not public, so the
CSVs are not in this repo; the script is here to show how the figures were made.

usage: python tools/hardware_figures.py PATH_TO_FIGURE_CSVS
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt
S = (sys.argv[1] if len(sys.argv) > 1 else "hardware_data") + "/"
OUT = "media/"
LOCKED = dict(color="0.35", linestyle="--", linewidth=1.3)
CTRL = dict(color="#0072B2", linestyle="-", linewidth=1.6)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 200})
def letters(axes, x=-0.08, y=1.02):
    for l, ax in zip("abcdef", axes):
        ax.text(x, y, f"({l})", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")

# 1. one real walking trial
d = pd.read_csv(S + "F2_example_walking_trial_data.csv")
t = d.time_s.values
win = (t >= 10) & (t <= 20)
fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.6), sharex=True, layout="constrained")
for ax, ax_name, lab in zip(axes, "xyz", ["forward [mm]", "left [mm]", "up [mm]"]):
    lk = d[f"locked_{ax_name}_mm"].values; lk = medfilt(lk - lk.mean(), 11)
    ee = d[f"ee_minus_goal_{ax_name}_mm"].values
    ee = ee - ee[(t >= 5)].mean()
    ax.plot(t[win], lk[win], label="arm locked to the mount (computed)", **LOCKED)
    ax.plot(t[win], ee[win], label="arm under control (measured, Vicon)", **CTRL)
    ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
    ax.set_ylabel(lab)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
axes[-1].set_xlabel("time in trial [s]")
fig.savefig(OUT + "hw_walking_trial.png"); plt.close(fig)

# 2. rejection by speed and by frequency
f4 = pd.read_csv(S + "F4_motion_and_attenuation_by_speed_data.csv")
f5 = pd.read_csv(S + "F5_attenuation_vs_frequency_data.csv")
pooled = [p for p in f4.participant.unique() if p != "P1"]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.35]))
ax = axes[0]
speeds = [0.5, 1.0, 1.5]
for p in pooled:
    s = f4[(f4.participant == p) & f4.speed_m_s.isin(speeds)].sort_values("speed_m_s")
    ax.plot(s.speed_m_s, 100 * (1 - s.attenuation_ratio), color="0.75", linewidth=0.9, marker="o", markersize=3)
gm = [100 * (1 - np.exp(np.log(f4[(f4.speed_m_s == v) & f4.participant.isin(pooled)].attenuation_ratio).mean())) for v in speeds]
ax.plot(speeds, gm, color=CTRL["color"], linewidth=2.2, marker="o", markersize=6, label="all participants")
for v, g in zip(speeds, gm):
    ax.annotate(f"{g:.0f} %", (v, g), textcoords="offset points", xytext=(9, -14), ha="left", fontsize=9, color=CTRL["color"])
ax.set_ylim(0, 100); ax.set_xticks(speeds); ax.set_xlim(0.35, 1.65)
ax.set_xlabel("treadmill speed [m/s]"); ax.set_ylabel("mount motion removed [%]")
ax.plot([], [], color="0.75", marker="o", markersize=3, linewidth=0.9, label="one participant")
ax.legend(loc="lower left")
ax = axes[1]
g = f5[f5.participant.isin(pooled) & (f5.freq_hz <= 2.0)]
med = g.groupby("freq_hz").gain.median()
mod = g.groupby("freq_hz")[["model_logged", "model_kd0", "model_delayed"]].median()
ax.plot(med.index, med.values, color=CTRL["color"], linewidth=2.2, label="measured (median of 6)")
ax.plot(mod.index, mod.model_logged, color="0.35", linestyle="--", linewidth=1.3, label="model, velocity term on time")
ax.plot(mod.index, mod.model_delayed, color="#D55E00", linewidth=1.4, label="model, velocity signal 65-90 ms late")
ax.axhline(1.0, color="0.6", linestyle=":", linewidth=1.0)
ax.text(2.0, 1.02, "no benefit", fontsize=8.5, color="0.45", va="bottom", ha="right")
ax.set_xscale("log"); ax.set_xticks([0.125, 0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2"])
ax.set_ylim(0, 1.12); ax.set_xlim(0.12, 2.05)
ax.set_xlabel("frequency [Hz], walking at 1.0 m/s"); ax.set_ylabel("motion left / motion imposed")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
letters(axes, x=-0.12)
fig.savefig(OUT + "hw_rejection.png"); plt.close(fig)
print("mount motion removed [%]:", [round(g) for g in gm])
