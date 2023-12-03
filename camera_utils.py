import numpy as np
import torch


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
    z_axis = np.cross(x_axis, y_axis)

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    assert np.dot(x_axis, y_axis) < 1e-4, "x and y axes are not orthogonal"
    assert np.dot(x_axis, z_axis) < 1e-4
    assert np.dot(y_axis, z_axis) < 1e-4

    return np.array([x_axis, y_axis, z_axis]).T


def get_torch3d_R_T(cam_frame_rotation, cam_pos):
    """ get extrinsic camera parameters in torch3d conventions from camera frame rotation and position"""
    # torch3d and mujoco have opposite z and x axes
    muj2torch = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # we invert the rotation by transposing the matrix, then move from mujoco to torch3d coordinates
    R = muj2torch @ cam_frame_rotation.T

    T = - R @ cam_pos

    return torch.tensor(R, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
