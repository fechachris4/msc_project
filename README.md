# MSc Project — SRL MuJoCo Simulation

MuJoCo simulation of a torso-mounted dual Kinova Gen3 (supernumerary robotic
limbs) setup, working toward torque NMPC "chicken-head" base-motion
stabilization.

## Setup

Requires the project virtual environment (Python 3.14 with `mujoco` and
`matplotlib`):

```bash
source .venv/bin/activate
```

If `.venv` is missing, recreate it from the Python.org framework install:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv --system-site-packages .venv
```

**macOS note:** anything that opens the MuJoCo viewer must be run with
`mjpython` (installed with the `mujoco` package), not plain `python` —
`launch_passive` needs the main thread on macOS.

## Running

Bare simulation (viewer only):

```bash
mjpython main.py
```

FK validation — live plot comparing the directly measured right end-effector
position against the FK-composed one (world → torso mocap → Kinova base → EE):

```bash
python -m plotting.fk_validation
```

Live plot while the sim runs headless (no MuJoCo viewer, so plain python
is fine — matplotlib needs the main thread on macOS). Close the plot window
to stop; the figure is saved to `plots/fk_validation.png`.

## Layout

```
main.py                    bare sim loop (viewer + mj_step)
sim/
  scene.xml                MJCF scene: torso + dual Kinova Gen3
  world.py                 model/data loading, cached body & site IDs
  assets/kinova_gen3/      vendored Kinova Gen3 model
controller/
  kinematics.py            frame transforms and FK for the right EE
  pd.py                    (placeholder) PD controller
plotting/
  fk_validation.py         direct-vs-FK EE validation plot, standalone
tests/
  test_kinematics.py       kinematics unit tests
```

## Conventions

- Frames: `T_A_B` denotes the pose of frame B expressed in frame A
  (see the docstring in `controller/kinematics.py`).
- Positions in meters, world frame unless stated otherwise.
- Each concern lives in its own module; scripts are runnable standalone via
  `python -m <package>.<module>`.
