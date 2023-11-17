import numpy as np
from camera_simulator import CameraSimulator
import matplotlib.pyplot as plt
import time
from camera_utils import homogeneous_to_cartesian, cartesian_to_homogeneous, intrinsic_matrix_from_params, \
    extrinsic_matrix_from_rotation_translatin, xy_axes_to_frame_rotation
from configurations import res_x, res_y, fov_x, fov_y, cam_1_pos, cam_1_xy_axes, table_intresest_points



if __name__ == '__main__':
    intrinsic_matrix = intrinsic_matrix_from_params(res_x, res_y, fov_x, fov_y)

    # set of parameters for extrinsics, camera at the corner looking down to the table:
    cam_pos = cam_1_pos
    cam_frame_rotation = xy_axes_to_frame_rotation(cam_1_xy_axes[0], cam_1_xy_axes[1])

    extrinsic_matrix = extrinsic_matrix_from_rotation_translatin(cam_frame_rotation, cam_pos)

    camera_matrix = intrinsic_matrix @ extrinsic_matrix

    interest_points = table_intresest_points
    interest_points = [cartesian_to_homogeneous(p) for p in interest_points]
    interest_points = np.array(interest_points).T

    interest_points_im_plane = camera_matrix @ interest_points
    interest_points_im_plane = np.array([homogeneous_to_cartesian(corner) for corner in interest_points_im_plane.T])

    # TODO: use 4x4 matrix for extrinsics to check if target is at FOV
    cam_sim = CameraSimulator(resolution=(res_x, res_y), fovy=fov_y, launch_viewer=False,
                              world_file="./data/world_table.xml")
    res = cam_sim.render(rotation_matrix=cam_frame_rotation, position=cam_pos)

    plt.imshow(res)
    plt.plot(interest_points_im_plane[:, 0], interest_points_im_plane[:, 1], 'o', color='red')
    plt.show()

    time.sleep(1)

    print("---data:---")
    print("camera_frame_rotation:\n", cam_frame_rotation)
    print("camera_position:\n", cam_pos)
    print("resx: ", res_x, "resy: ", res_y, "fovx: ", fov_x, "fovy: ", fov_y)
    print("interest_points:\n", interest_points.squeeze())

    print("---results:---")
    print("intrinsic matrix:\n", intrinsic_matrix)
    print("extrinsic matrix:\n", extrinsic_matrix)
    print("camera matrix:\n", camera_matrix)
    print("interest points in image plane:\n", interest_points_im_plane)
