from controller import servo
from plotting.position_error import run

if __name__ == "__main__":
    run("right", servo.right_pose_error)
