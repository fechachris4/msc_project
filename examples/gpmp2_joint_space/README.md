# GPMP2 joint-space planning demonstration

This folder is deliberately separate from the live Python controller. It
shows the missing high-level planning layer without sending commands to
MuJoCo or the robot.

`main.cpp` optimizes a sequence of joint position and joint velocity states:

```text
(q0, qdot0), (q1, qdot1), ... (qN, qdotN)
```

The factor graph contains:

- fixed start and goal joint configurations;
- zero start and end velocity;
- a Gaussian-process smoothness prior between adjacent states;
- joint-position and joint-velocity limit costs;
- arm-sphere versus human-cylinder SDF costs at every support state;
- the same collision costs at four GP-interpolated states between each pair.

The output is a dense CSV joint trajectory. The program checks the start and
goal errors, joint-position limits, joint-velocity limits, and minimum dense
sphere clearance before printing it. A result that retains excessive soft
factor error is rejected rather than treated as executable.

## Build on the Ubuntu GPMP2 machine

GPMP2 and GTSAM must be native libraries for that machine:

```bash
cmake -S examples/gpmp2_joint_space \
      -B build/gpmp2_joint_space \
      -DGPMP2_ROOT=/usr/local
cmake --build build/gpmp2_joint_space --parallel
```

Run the built-in illustrative start and goal:

```bash
./build/gpmp2_joint_space/gpmp2_joint_space_demo \
  > /tmp/gpmp2_joint_trajectory.csv
```

Or provide seven start joints followed by seven goal joints, all in radians:

```bash
./build/gpmp2_joint_space/gpmp2_joint_space_demo \
  q1s q2s q3s q4s q5s q6s q7s \
  q1g q2g q3g q4g q5g q6g q7g \
  > /tmp/gpmp2_joint_trajectory.csv
```

Diagnostics and planning failures go to standard error, so they do not
contaminate the CSV.

## Important boundary

This is genuine GPMP2/GTSAM factor-graph code, but it is an educational
standalone example—not a validated production planner. Its three small
link-origin spheres only make the GPMP2 collision-factor data flow visible;
they are not a collision model of the actual arm. Before integration, they
must be replaced by a frame-validated GPMP2 mapping of all current
MuJoCo-derived spheres, and every dense waypoint must be replayed against
MuJoCo ground truth.

The current Mac cannot link this example: its copied `libgpmp2.so` and
`libgtsam.so` files are Linux x86-64 binaries. The source can be syntax-checked
against the copied headers locally, but execution belongs on the Ubuntu
GPMP2 installation.
