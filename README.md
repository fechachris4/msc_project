# Holding a robot's hand still while its wearer moves

[![tests](https://github.com/fechachris4/msc_project/actions/workflows/tests.yml/badge.svg)](https://github.com/fechachris4/msc_project/actions/workflows/tests.yml)

Christian Akabueze · MSc Human and Biological Robotics, Imperial College London (MUVE Lab) · 2026

A robot arm you wear is an extra pair of hands, until you take a step. Every step moves the mount the arms sit on, and anything they hold moves with it. My MSc project asked how much of that motion two worn Kinova Gen3 arms can cancel, so their end-effectors (the robot's hands) stay fixed in the room rather than on the body.

<p align="center"><img src="media/hw_turn.gif" width="480" alt="The wearer twists his torso; the end-effectors move much less"></p>

*Phone video from the lab, stabilised on the background. The controller is in world hold during setup (not a recorded trial). The wearer twists his torso by about 60°; the elbows swing, while the end-effectors move much less than the torso.*

Seven participants walked on a treadmill while the arms held a fixed point in the room. At a normal walking speed (1.0 m/s) the arms removed 63% of the mount's motion, averaged over six participants. Slow sway was mostly removed. The faster bounce that comes with each step largely was not.

The likely cause is timing. The arm learns how the mount is moving 61-71 ms late, and in simulation a delay of that size is enough to cancel the benefit of using that signal. So the next thing to improve on hardware is the velocity estimate.

![The rig: a participant walking on the treadmill with both arms on, and the same rig from the front](media/rig.jpg)

*The rig. Left: a participant walking on the treadmill with both arms on. Right: the same rig from the front. Vicon cameras on the truss and on tripods track the mount and the end-effectors.*

## What is in this repo

The MuJoCo simulation and controller I built for the project, a C++20 port of them in [`cpp/`](cpp/README.md), and the scripts behind the hardware figures. The controller that ran on the robot lives on the MUVE Lab machine and is not public, and neither are the participant data. The arm model is from MuJoCo Menagerie; the controller, safety filter, experiments and C++ port are mine.

The thesis is "World-Stable Supernumerary Effectors for Human Augmentation Under Locomotion-Scale Base Motion". It is under examination, and I will link it once it is marked. The hardware numbers here match it.

## Read more

- [Hardware trials](docs/hardware-trials.md): how a trial is scored, results by speed and frequency, and what limits the arm.
- [Simulation](docs/simulation.md): the delay experiment, how the controller works, and the frequency sweep.
- [Scope and next steps](docs/scope-and-next-steps.md): what these results cover and what comes next.
- [Code map](docs/code-map.md): where things live in the repo.

## Run the simulation

Tested on Python 3.14.4 with `mujoco`, `pin`, `numpy`, `scipy`, `osqp`, `matplotlib`, and `pandas` for the hardware figure scripts (`requirements.txt`; pinned versions in `requirements-lock.txt`). Run from the repo root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mjpython tools/mount_disturbance.py both   # viewer at f = 1.8 Hz (mjpython on macOS)
python tools/velocity_delay.py             # delay figure, about 4 minutes
python tools/disturbance_freq_sweep.py     # results table and sweep figure
python tools/feedforward_compare.py right --f=1.8
python tools/make_readme_media.py          # the simulation video
python -m unittest discover tests
```

Gains, limits, targets and the safety envelope are in `config/control.toml`. The Kinova Gen3 model in `sim/assets/kinova_gen3/` is from MuJoCo Menagerie (BSD licence, Kinova). My code is MIT licensed.

## Contact

For questions about the project or the code, [open an issue](https://github.com/fechachris4/msc_project/issues) or email fecha412@gmail.com. The repository is maintained by Christian Akabueze.
