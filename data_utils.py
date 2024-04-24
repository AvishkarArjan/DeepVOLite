import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pykitti
from config import *
import numpy as np
import math
import sys

kittiTransform = transforms.Compose([transforms.ToTensor()])
normalizer = transforms.Compose(
    [
        transforms.Normalize(
            (0.19007764876619865, 0.15170388157131237, 0.10659445665650864),
            (0.2610784009469139, 0.25729316928935814, 0.25163823815039915),
        )
    ]
)
# epsilon value for euler transformation
EPS = np.finfo(float).eps * 4.0

def euler_from_matrix(matrix):
    # Rotation matrix to euler angles
    # y-x-z Tait–Bryan angles intrincic
    # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    i = 2
    j = 0
    k = 1
    frame = 1

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
   
    cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
    if cy > EPS:
        ax = math.atan2(M[k, j], M[k, k])
        ay = math.atan2(-M[k, i], cy)
        az = math.atan2(M[j, i], M[i, i])
    else:
        ax = math.atan2(-M[j, k], M[j, j])
        ay = math.atan2(-M[k, i], cy)
        az = 0.0

    if frame:
        ax, az = az, ax
    return ax, ay, az


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def get_translation(mat):
    t = mat[:, -1][:-1]
    return t

def get_rotation(mat):
    return mat[:3, :3]

# due to -pi to pi discontinuity
def normalize_angle_delta(angle):
    if angle > np.pi:
        angle = angle - 2 * np.pi
    elif angle < -np.pi:
        angle = 2 * np.pi + angle
    return angle


class KittiOdomDataset(Dataset):
    def __init__(self, seq, path, transform=kittiTransform , normalizer=normalizer):

        self.odom = pykitti.odometry(path, seq)
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len(self.odom)-1- SEQ_LENGTH

    def __getitem__(self, idx):

        """get original rgb_img, pos, angles"""

        imgs = []
        trans = [] # these are the pos (positions ?)
        angle = []

        # this method seems a little different from previous method
        original_angle = torch.FloatTensor(euler_from_matrix(self.odom.poses[idx])) # Get euler angles
        original_trans = torch.FloatTensor(get_translation(self.odom.poses[idx])) # Extract translation component from 4x4 pose
        original_rot = get_rotation(self.odom.poses[idx]).T

        trans.append(original_trans)
        angle.append(original_angle)

        for i in range(SEQ_LENGTH):
            cur_img = self.transform(self.odom.get_rgb(idx+i)[0]) - 0.5
            next_img = self.transform(self.odom.get_rgb(idx+i)[0]) - 0.5
            cur_img = self.normalizer(cur_img)
            next_img = self.normalizer(next_img)


            next_angle = torch.FloatTensor(euler_from_matrix(self.odom.poses[idx+i+1]))
            next_trans = torch.FloatTensor(get_translation(self.odom.poses[idx+i+1]))

            imgs.append(torch.cat((cur_img, next_img), dim=0))
            trans.append(next_trans)
            angle.append(next_angle)

        # A sequence of 6 combined frames
        imgs = torch.stack(imgs)
        trans = torch.stack(trans)
        angle = torch.stack(angle)

        # Preprocessing
        trans[1:] = trans[1:] - original_trans
        angle[1:] = angle[1:] - original_angle

        for i in range(1, len(trans)):
            loc = torch.FloatTensor(original_rot.dot(trans[i]))
            trans[i][:] = loc[:]

        trans[2:] = trans[2:] - trans[1:-1]
        angle[2:] = angle[2:] - angle[1:-1]

        for i in range(1, len(angle)):
            angle[i][0] = normalize_angle_delta(angle[i][0])
            angle[i][1] = normalize_angle_delta(angle[i][1])
            angle[i][2] = normalize_angle_delta(angle[i][2])
        # print("dataset:",imgs.shape)
        # print("dataset:",trans.shape)
        # print("dataset:",angle.shape)

        # print(sys.getsizeof(imgs))
        # print(sys.getsizeof(trans))
        # print(sys.getsizeof(angle))
       
        return imgs, trans, angle

        



