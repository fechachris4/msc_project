"""media/hw_replay.gif: one real treadmill trial (Vicon data) replayed at real
speed. Reads the same figure-data CSV as hardware_figures.py, which is not in
this repo (participant data are not public).

usage: python tools/hardware_replay.py PATH_TO_FIGURE_CSVS
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import medfilt

S = (sys.argv[1] if len(sys.argv) > 1 else "hardware_data") + "/F2_example_walking_trial_data.csv"
OUT = "media/hw_replay.gif"
GREY, BLUE = "0.45", "#0072B2"
T0, T1, FPS, TRAIL = 10.0, 20.0, 10, 1.0

d = pd.read_csv(S)
t = d.time_s.values
steady = t >= 5          # centre both traces on the same span, as in hardware_figures.py
def locked(a):           # computed point is noisier; 11-frame median filter, display only
    v = d[f"locked_{a}_mm"].values
    return medfilt(v - v[steady].mean(), 11)
def ee(a):
    v = d[f"ee_minus_goal_{a}_mm"].values
    return v - v[steady].mean()
L = {a: locked(a) for a in "xyz"}
E = {a: ee(a) for a in "xyz"}

frames = np.arange(T0, T1, 1.0 / FPS)
idx = np.searchsorted(t, frames)
tr = int(TRAIL * 100)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), dpi=130, layout="constrained")
views = [("y", "z", "seen from behind", "left-right [mm]"), ("x", "z", "seen from the side", "forward-back [mm]")]
art = []
for ax, (h, v, title, xl) in zip(axes, views):
    ax.set_xlim(-110, 110); ax.set_ylim(-70, 70); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color="0.25")
    ax.set_xlabel(xl); ax.set_ylabel("up-down [mm]")
    ax.axhline(0, color="0.9", lw=0.6, zorder=0); ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    lt, = ax.plot([], [], color=GREY, lw=1.2, ls="--", alpha=0.8)
    ld, = ax.plot([], [], "o", ms=9, mfc="white", mec=GREY, mew=2)
    et, = ax.plot([], [], color=BLUE, lw=1.6, alpha=0.9)
    ed, = ax.plot([], [], "o", ms=9, color=BLUE)
    art.append((h, v, lt, ld, et, ed))
axes[0].plot([], [], "o", ms=8, mfc="white", mec=GREY, mew=2, ls="--", color=GREY, label="arm locked to the mount")
axes[0].plot([], [], "o", ms=8, color=BLUE, ls="-", label="arm under control")
fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=9.5)
clock = fig.text(0.995, 0.995, "", ha="right", va="top", fontsize=9, color="0.4", family="monospace")

def draw(k):
    i = idx[k]; j = max(0, i - tr)
    out = []
    for h, v, lt, ld, et, ed in art:
        lt.set_data(L[h][j:i + 1], L[v][j:i + 1]); ld.set_data([L[h][i]], [L[v][i]])
        et.set_data(E[h][j:i + 1], E[v][j:i + 1]); ed.set_data([E[h][i]], [E[v][i]])
        out += [lt, ld, et, ed]
    clock.set_text(f"t = {t[i]:4.1f} s")
    return out + [clock]

anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(OUT, writer=PillowWriter(fps=FPS))
print(OUT)
