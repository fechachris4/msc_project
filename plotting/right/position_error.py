from controller import pd
from plotting.position_error import run

if __name__ == "__main__":
    run("right", pd.right_pose_error)
