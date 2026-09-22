"""README video: arms locked vs reactive vs reactive + feedforward.

Same scripted six-axis mount disturbance as disturbance_freq_sweep.py at
F_HZ, unscaled amplitudes. The controlled runs are recorded as MuJoCo states
and re-rendered; the locked panel replays the same mount motion with the
joints frozen at their settled pose (the no-control counterfactual). The
readout is the running RMS of the worse arm's position error since onset.

Outputs: media/hold_pose.gif, media/hold_pose.mp4

usage: python tools/make_readme_media.py [--f=1.8]
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import disturbance_freq_sweep as sweep  # noqa: E402
import report_style  # noqa: E402
import disturbance_run  # noqa: E402
import mount_disturbance  # noqa: E402
from controller import servo  # noqa: E402
from sim import world  # noqa: E402

OUT = Path("media")
PANEL = (320, 150)          # w, h of each wide (context) render
CLOSE = (320, 200)          # w, h of the close-up on the left target
SHOW_S = 2.3                # two pattern periods at 1.8 Hz
LEAD_S = 0.3                # static mount shown before onset
FPS = 25
BG = (24, 26, 32)
FG = (230, 230, 230)
COLORS = {"locked": "#9a9a9a", "reactive": report_style.REACTIVE["color"],
          "ff": report_style.FEEDFORWARD["color"]}
TITLES = {"locked": "arms locked (no control)", "reactive": "reactive",
          "ff": "reactive + feedforward"}
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
FONT_BOLD = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"


class _StateRecorder:
    """Stands in for mujoco.Renderer inside disturbance_run.run: keeps states."""

    snapshots = []

    def __init__(self, model, height, width):
        pass

    def update_scene(self, data, camera=None):
        _StateRecorder.snapshots.append(
            (data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy()))

    def render(self):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        pass


def record(ff_enabled):
    config = dataclasses.replace(
        servo.CONTROL, velocity_feedforward_enabled=ff_enabled)
    real_renderer = mujoco.Renderer
    _StateRecorder.snapshots = []
    mujoco.Renderer = _StateRecorder
    try:
        settled, _, rows, _ = disturbance_run.run(
            world.SIDES, 1.0, sweep.AMPLITUDE_KEY, record_gif=True,
            controller_config=config)
    finally:
        mujoco.Renderer = real_renderer
    assert settled, "arms did not settle before the disturbance"
    return rows, list(_StateRecorder.snapshots)


def worst_error_mm(rows, key):
    t = np.array([r["t"] for r in rows])
    err = np.max([1e3 * np.linalg.norm(
        np.array([r[s][key] for r in rows]), axis=1) for s in world.SIDES],
        axis=0)
    return t, err


def _camera(lookat, distance, azimuth=-140.0, elevation=-22.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


_OPT = mujoco.MjvOption()
_OPT.sitegroup[:] = 0        # hide marker sites (frame origins)


def _draw(renderer, cam):
    renderer.update_scene(world.data, camera=cam, scene_option=_OPT)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    return Image.fromarray(renderer.render().copy())


def running_rms(t, err, now):
    """RMS of err over [0, now]; zero before the disturbance starts."""
    window = (t >= 0.0) & (t <= now)
    return float(np.sqrt(np.mean(err[window] ** 2))) if window.any() else 0.0


def render_states(snapshots, wide, close, lock_q=None):
    """Wide view plus a close-up fixed on the (world-fixed) left target."""
    target = world.model.body("left_target").mocapid[0]
    wide_cam = _camera([0.18, 0.04, 1.26], 1.15, elevation=-32.0)
    images = []
    for qpos, mocap_pos, mocap_quat in snapshots:
        world.data.qpos[:] = qpos if lock_q is None else lock_q
        world.data.mocap_pos[:] = mocap_pos
        world.data.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(world.model, world.data)
        close_cam = _camera(mocap_pos[target], 0.34, elevation=-30.0)
        images.append((_draw(wide, wide_cam), _draw(close, close_cam)))
    return images


def strip_chart(series, width, height):
    """Error-vs-time strip; returns the image and a t -> x pixel map."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    bg = tuple(c / 255 for c in BG)
    fg = tuple(c / 255 for c in FG)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for name, (t, err) in series.items():
        ax.plot(t, err, color=COLORS[name], linewidth=2.0,
                linestyle="--" if name == "locked" else "-",
                label=TITLES[name])
    leg = ax.legend(loc="upper right", ncol=3, fontsize=11, frameon=False,
                    handlelength=1.8, borderaxespad=0.1)
    for text in leg.get_texts():
        text.set_color(fg)
    ax.set_xlim(-LEAD_S, SHOW_S)
    ax.set_ylim(0, 1.55 * max(err.max() for _, err in series.values()))
    ax.set_xlabel("time since disturbance onset [s]", color=fg, fontsize=12)
    ax.set_ylabel("error [mm]", color=fg, fontsize=12)
    ax.tick_params(colors=fg, labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(fg)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.97, bottom=0.27)
    fig.canvas.draw()
    image = Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba()).convert("RGB")
    to_px = ax.transData

    def x_of(t):
        return to_px.transform((t, 0))[0]

    y0, y1 = (height - to_px.transform((0, v))[1]
              for v in ax.get_ylim())
    plt.close(fig)
    return image, x_of, (y1, y0)


def compose(panels, chart, x_of, y_span, t, readouts):
    w, h = PANEL
    ch = CLOSE[1]
    title_h, chart_h = 34, chart.size[1]
    canvas = Image.new("RGB", (3 * w, title_h + h + ch + chart_h), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_BOLD), 16)
    big_font = ImageFont.truetype(str(FONT_BOLD), 22)
    small_font = ImageFont.truetype(str(FONT), 13)
    for i, (name, (image, inset)) in enumerate(panels.items()):
        x0 = i * w
        canvas.paste(image, (x0, title_h))
        canvas.paste(inset, (x0, title_h + h))
        draw.line([(x0, title_h + h), (x0 + w, title_h + h)],
                  fill=BG, width=3)
        draw.text((x0 + w / 2, title_h / 2), TITLES[name], fill=COLORS[name],
                  font=title_font, anchor="mm")
        draw.text((x0 + 12, title_h + h + ch - 34), "RMS so far",
                  fill=FG, font=small_font, anchor="ls",
                  stroke_width=2, stroke_fill=BG)
        draw.text((x0 + 12, title_h + h + ch - 10), f"{readouts[name]:4.1f} mm",
                  fill=COLORS[name], font=big_font, anchor="ls",
                  stroke_width=3, stroke_fill=BG)
    for i in (1, 2):
        draw.line([(i * w, title_h), (i * w, title_h + h + ch)],
                  fill=BG, width=3)
    canvas.paste(chart, (0, title_h + h + ch))
    x = x_of(t)
    top, bottom = (title_h + h + ch + v for v in y_span)
    draw.line([(x, top), (x, bottom)], fill=FG, width=1)
    return canvas


def main(argv):
    f_hz = mount_disturbance.pop_float_option(argv, "f", 1.8)
    sweep._fundamental_hz[0] = f_hz
    mount_disturbance.disturbance_params = sweep.fixed_amplitude_params
    OUT.mkdir(exist_ok=True)

    rows_r, snaps_r = record(ff_enabled=False)
    rows_f, snaps_f = record(ff_enabled=True)
    series = {"locked": worst_error_mm(rows_r, "rigid_err"),
              "reactive": worst_error_mm(rows_r, "e_pos"),
              "ff": worst_error_mm(rows_f, "e_pos")}
    for name, (t, err) in series.items():
        on = t >= 0.0
        print(f"{name:9s} worst-arm EE error RMS "
              f"{np.sqrt(np.mean(err[on] ** 2)):.1f} mm")

    # Frame k of the recording is at t = -GIF_LEAD_S + k * GIF_FRAME_S.
    step = disturbance_run.GIF_FRAME_S
    first = int(round((disturbance_run.GIF_LEAD_S - LEAD_S) / step))
    count = int(round((LEAD_S + SHOW_S) / step))
    keep = slice(first, first + count)
    times = -disturbance_run.GIF_LEAD_S + step * np.arange(len(snaps_r))[keep]

    world.model.vis.global_.offwidth = max(PANEL[0],
                                           world.model.vis.global_.offwidth)
    world.model.vis.global_.offheight = max(PANEL[1],
                                            world.model.vis.global_.offheight)
    wide = mujoco.Renderer(world.model, PANEL[1], PANEL[0])
    close = mujoco.Renderer(world.model, CLOSE[1], CLOSE[0])
    lock_q = snaps_r[first][0]
    panels = {
        "locked": render_states(snaps_r[keep], wide, close, lock_q=lock_q),
        "reactive": render_states(snaps_r[keep], wide, close),
        "ff": render_states(snaps_f[keep], wide, close),
    }
    wide.close()
    close.close()

    chart, x_of, y_span = strip_chart(series, 3 * PANEL[0], 170)
    frames = []
    for k, t in enumerate(times):
        readouts = {name: running_rms(*series[name], t) for name in series}
        frames.append(compose({n: panels[n][k] for n in panels},
                              chart, x_of, y_span, t, readouts))

    gif = OUT / "hold_pose.gif"
    palette = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    small = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"gif: {gif} ({gif.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    tmp = OUT / "_frames"
    tmp.mkdir(exist_ok=True)
    for k, frame in enumerate(frames):
        frame.save(tmp / f"{k:04d}.png")
    mp4 = OUT / "hold_pose.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(FPS), "-i", str(tmp / "%04d.png"), "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "20", str(mp4)], check=True)
    for png in tmp.glob("*.png"):
        png.unlink()
    tmp.rmdir()
    print(f"mp4: {mp4}")


if __name__ == "__main__":
    main(sys.argv[1:])
