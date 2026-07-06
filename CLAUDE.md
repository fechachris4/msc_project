# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc research project simulating a dual-arm Kinova Gen3 manipulator mounted on a torso base in MuJoCo. The current milestone (per `tasks/todo.md`) is a torso-box + dual-Gen3 base scene with a working passive-viewer simulation loop; `controller/pd.py` is a placeholder for the PD/torque controller work planned next.

## Commands

Use the project virtualenv (Python 3.14 with `mujoco` and `matplotlib`; see README for how to recreate it):

```bash
source .venv/bin/activate
```

Run the simulation (passive viewer, opens a window). On macOS anything that opens the MuJoCo viewer must use `mjpython`, not plain `python`:

```bash
mjpython main.py
```

Run the FK validation plot (headless sim, plain `python` is fine):

```bash
python -m plotting.fk_validation
```

Run the tests:

```bash
python -m unittest tests.test_kinematics
```

Validate MJCF changes without launching MuJoCo:

```bash
xmllint --noout sim/scene.xml
```

## Architecture

- **`sim/scene.xml`** — top-level MJCF scene. It does *not* define the robot geometry itself; it `<attach>`es two instances of the vendored `assets/kinova_gen3/gen3.xml` model (with `right_`/`left_` prefixes) onto a single mocap `torso` body, at mirrored mount frames (`axisangle` flipped in sign left vs. right). The torso and two target spheres are all `mocap="true"` bodies — they are kinematically driven (e.g. by a controller or scripted motion), not simulated as free/jointed bodies, so there is currently no base velocity state to feed a controller.
- **`<contact><exclude>` block** — required because attaching two arms to a mocap-welded torso disables MuJoCo's automatic parent-child contact filtering at the mount points; each arm's `base_link`↔`shoulder_link` pair is explicitly excluded to prevent spurious self-collisions. If you add more attachment points or bodies near the mounts, check whether new exclusions are needed.
- **`sim/assets/kinova_gen3/`** — vendored (not a git submodule) copy of the MuJoCo Menagerie Kinova Gen3 MJCF description, kept in-tree for portability across machines. Its own `README.md` documents how it was derived from the URDF and details specific to mounting a gripper (e.g. Robotiq 2F-85 offsets) if that's ever added. Don't hand-edit vendored files casually; prefer changes in `sim/scene.xml`.
- **`main.py`** — minimal MuJoCo passive-viewer loop: load model → step physics → sync viewer → sleep to hold real-time rate via `model.opt.timestep`. This is the baseline harness that any controller (`controller/pd.py`) will hook into by writing to `data.ctrl` / mocap poses between `mj_step` calls.
- **`controller/transforms.py`** — pure rigid-transform math (SE(3), quaternion/axis-angle/RPY rotations). NumPy only; reusable anywhere.
- **`controller/kinematics.py`** — `extract_chain(model, base_body, ee_site)` reads hinge-chain constants from `MjModel` once; `fk(chain, qpos)` is the analytical FK and never reads `MjData`. Validated against MuJoCo at random configurations in `tests/test_kinematics.py`.
- **`controller/frames.py`** — scene-specific glue (deliberately disposable): the two arm chains, torso mounts, and the world-frame composition `T_W_E = T_W_T · T_T_K · T_K_E(q)` (world → torso mocap → Kinova base → EE site).
- **`sim/targets.py`** — set/read the EE target positions (mocap spheres, world frame, meters).
- **`plotting/fk_validation.py`** — standalone headless live plot comparing direct vs. FK-composed right EE position; saves to `plots/fk_validation.png`.
- **`controller/pd.py`** — currently empty; intended location for the PD/torque controller driving the arms (and eventually torso base-motion feedforward, per project notes).

## MJCF conventions in this repo (from `tasks/lessons.md`)

- Keep magic-number vectors (e.g. keyframe `qpos`/`ctrl` blobs) out of hand-edited MJCF — compose them at load time in Python from named sources instead. If a numeric literal must live in XML, comment where each number came from.
- For asset/model load failures, first confirm the referenced file path actually exists on disk before assuming an XML syntax error; then re-validate with `xmllint --noout` and a MuJoCo `compile` of the parent scene.
