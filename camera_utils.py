import numpy as np
import torch
from scipy.spatial.transform import Rotation


def homogeneous_to_cartesian(homogeneous):
    return homogeneous[0:-1] / homogeneous[-1]


def cartesian_to_homogeneous(cartesian):
    return np.append(cartesian, 1)


def intrinsic_matrix_from_params(res_x, res_y, fov_x, fov_y):
    focal_x = res_x / (2 * np.tan(fov_x / 2 * np.pi / 180))
    focal_y = res_y / (2 * np.tan(fov_y / 2 * np.pi / 180))
    c_x = res_x / 2
    c_y = res_y / 2
    return np.array([[focal_x, 0, c_x],
                     [0, focal_y, c_y],
                     [0, 0, 1]])


def extrinsic_matrix_from_rotation_translatin(rotation_matrix, translation):
    extrinsic = np.zeros((3, 4))
    # rotation matrix is the rotation of the camera frame. The camera matrix is mapping from world to camera thus it
    # should be the inverse of the rotation matrix because it rotates back.

    x, y, z = rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]
    # rotation_matrix_ = rotation_matrix.copy()
    rotation_matrix_ = np.concatenate((-x.reshape(3, 1), y.reshape(3, 1), z.reshape(3, 1)), axis=1)

    translation_ = translation.copy()
    # translation_[0] = -translation_[0]

    extrinsic[0:3, 0:3] = rotation_matrix_.T
    extrinsic[0:3, 3] = -rotation_matrix_.T @ translation_
    return extrinsic


def xy_axes_to_frame_rotation(x_axis, y_axis):
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    # reduce non orthogonal part of y and x:
    y_axis = y_axis - y_axis.T @ x_axis * x_axis

    z_axis = np.cross(x_axis, y_axis)

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    assert np.dot(x_axis, y_axis) < 1e-5, "x and y axes are not orthogonal"
    assert np.dot(x_axis, z_axis) < 1e-5
    assert np.dot(y_axis, z_axis) < 1e-5

    return np.array([x_axis, y_axis, z_axis]).T


def get_torch3d_R_T(cam_frame_rotation, cam_pos):
    """ get extrinsic camera parameters in torch3d conventions from camera frame rotation and position"""
    # torch3d and mujoco have opposite z and x axes
    muj2torch = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # we invert the rotation by transposing the matrix, then move from mujoco to torch3d coordinates
    R_temp = muj2torch @ cam_frame_rotation.T
    T = - R_temp @ cam_pos

    # torch 3d uses euler angles in the order ZYX, mujoco uses XYZ, in addition rotation around Y axis is opposite in
    # pytorch 3d. For some reason in order to compute the T vector we need to use the non-inverted rotation and order R.
    # Trust me, I had some bad time figuring this out.
    r = Rotation.from_matrix(cam_frame_rotation)
    euler = r.as_euler('xyz', degrees=True)
    euler_new = np.array([euler[2] + 180, -euler[1], euler[0]])
    cam_frame_rotation = Rotation.from_euler('zyx', euler_new, degrees=True).as_matrix()
    R = muj2torch @ cam_frame_rotation.T

    return torch.tensor(R, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
