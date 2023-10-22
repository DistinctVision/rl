import typing as tp
from dataclasses import dataclass

import math
import numpy as np
import torch

from rlbot.utils.structures.game_data_struct import Rotator


def rotation_to_quaternion(m: tp.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    trace = np.trace(m)
    q = np.zeros(4)

    if trace > 0:
        s = (trace + 1) ** 0.5
        q[0] = s * 0.5
        s = 0.5 / s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = (1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * inv_s
            q[3] = (m[2, 0] + m[0, 2]) * inv_s
            q[0] = (m[2, 1] - m[1, 2]) * inv_s
        elif m[1, 1] > m[2, 2]:
            s = (1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * inv_s
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * inv_s
            q[0] = (m[0, 2] - m[2, 0]) * inv_s
        else:
            s = (1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * inv_s
            q[2] = (m[1, 2] + m[2, 1]) * inv_s
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * inv_s

    # q[[0, 1, 2, 3]] = q[[3, 0, 1, 2]]

    return -q


def quat_to_rot_mtx(quat: tp.Union[tp.Tuple[float, float, float, float], np.ndarray, torch.Tensor]) -> np.ndarray:
    w = - quat[0]
    x = - quat[1]
    y = - quat[2]
    z = - quat[3]

    theta = np.zeros((3, 3))

    norm = np.dot(quat, quat)
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


@dataclass
class EulerAngles:
    
    pitch: float
    yaw: float
    roll: float
    
    @staticmethod
    def from_quaternion(q: tp.Union[tp.Tuple[float, float, float, float], np.ndarray, torch.Tensor]) -> 'EulerAngles':
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        sinp = 2 * (w * y - z * x)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        roll = np.arctan2(sinr_cosp, cosr_cosp)
        if abs(sinp) > 1:
            pitch = np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return EulerAngles(-pitch, yaw, -roll)
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'EulerAngles':
        return EulerAngles(arr[0], arr[1], arr[2])

    def __init__(self, pitch: tp.Union[float, 'EulerAngles', 'Rotator'] = 0,
                 yaw: float = 0,
                 roll: float = 0):

        if hasattr(pitch, 'pitch'):
            self.pitch = pitch.pitch
            self.yaw =  pitch.yaw
            self.roll = pitch.roll
        else:
            self.pitch = float(pitch)
            self.yaw = float(yaw)
            self.roll = float(roll)

    def __str__(self):
        return f"EulerAngles({self.pitch:.2f}, {self.yaw:.2f}, {self.roll:.2f})"

    def __repr__(self):
        return self.__str__()

    def to_array(self) -> tp.List[float]:
        return [self.pitch, self.yaw, self.roll]
    
    def to_rotation_matrix(self) -> np.ndarray:
        cp, cy, cr = math.cos(self.pitch), math.cos(self.yaw), math.cos(self.roll)
        sp, sy, sr = math.sin(self.pitch), math.sin(self.yaw), math.sin(self.roll)

        theta = np.zeros((3, 3))

        # front
        theta[0, 0] = cp * cy
        theta[1, 0] = cp * sy
        theta[2, 0] = sp

        # left
        theta[0, 1] = cy * sp * sr - cr * sy
        theta[1, 1] = sy * sp * sr + cr * cy
        theta[2, 1] = -cp * sr

        # up
        theta[0, 2] = -cr * cy * sp - sr * sy
        theta[1, 2] = -cr * sy * sp + sr * cy
        theta[2, 2] = cp * cr

        return theta
    
    def to_quaternion(self) -> np.ndarray:
        return rotation_to_quaternion(self.to_rotation_matrix())
    