import pykitti
import matplotlib.pyplot as plt
from config import *


def se3_to_position(mat):
    t = mat[:, -1][:-1]
    return t



def draw_gt(data_path ,seq):
    x = []
    y = []
    odom = pykitti.odometry(data_path, seq)
    for i in range(len(odom)):
        t = se3_to_position(odom.poses[i])
        x.append(t[0])
        y.append(t[2])
    plt.plot(x, y, color="g", label="ground truth")
    plt.show()

# def draw_route(y, y_hat, name, weight_folder=WEIGHT_FOLDER, c_y="r", c_y_hat="b"):
#     plt.clf()
#     x = [v[0] for v in y]
#     y = [v[2] for v in y]
#     plt.plot(x, y, color=c_y, label="ground truth")

#     x = [v[0] for v in y_hat]
#     y = [v[2] for v in y_hat]
#     plt.plot(x, y, color=c_y_hat, label="ground truth")
#     plt.savefig(f"{weight_folder}/{name}")
#     plt.gca().set_aspect("equal", adjustable="datalim")

if __name__== "__main__":
    draw_gt("00")
