from controller import servo
from plotting.position_error import run

if __name__ == "__main__":
    run("left", servo.left_pose_error)
