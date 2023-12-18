import torch
from geometric_mesh_segmentation import CameraParameters, get_mesh_segmentation_batch
from camera_utils import xy_axes_to_frame_rotation, get_torch3d_R_T
import matplotlib.pyplot as plt
import numpy as np
from camera_simulator import CameraSimulator
from utils import plot_segmentation_mask


# parameters:
mesh1_path = "./data/meshes/mug.obj"
scale = 0.02
table_height = 1.0

cam_1_pos = [0.5, -0.8, 1.2]
cam_1_xy_axes = [[0.685, 0.728, 0.000], [-0.487, 0.458, 0.743]]
cam_1_resx, cam_1_resy = 200, 200
cam_1_fov = 60
cam_1_znear, cam_1_zfar = 0.1, 100

cam_2_pos = np.array([-0.1,0.15,0.9])
cam_2_frame_rotation = np.array([[0.7071068, -0.7071068, 0.000], [0.7071068, 0.7071068, 0], [0, 0, 1]]).T # 45deg around Z
cam_2_resx, cam_2_resy = 200, 200
cam_2_fov = 45
cam_2_znear, cam_2_zfar = cam_1_znear, cam_1_zfar

cam_3_pos = [1, -1, 1.9]
cam_3_xy_axes = [[0.685, 0.728, 0.000], [-0.487, 0.458, 0.743]]
cam_3_resx, cam_3_resy = 200, 200
cam_3_fov = 45
cam_3_znear, cam_3_zfar = 0.1, 100

obj_position_1 = torch.Tensor([0.2, 0.3, 0.08 + table_height])
obj_orientation_1 = torch.Tensor([2.1, 0, 1.57])

obj_position_2 = torch.Tensor([-0.2, -0.1, 0.1 + table_height])
obj_orientation_2 = torch.Tensor([0.7, -0.7, 0.7])


def test_2_cameras_base_pose():
    cam_1_frame_rotation = xy_axes_to_frame_rotation(cam_1_xy_axes[0], cam_1_xy_axes[1])
    R1, T1 = get_torch3d_R_T(cam_1_frame_rotation, cam_1_pos)
    cam_params_1 = CameraParameters(res_x=cam_1_resx, res_y=cam_1_resy, fov=cam_1_fov, R=R1, T=T1,
                                    z_near=cam_1_znear, z_far=cam_1_zfar)

    R2, T2 = get_torch3d_R_T(cam_2_frame_rotation, cam_2_pos)
    cam_params_2 = CameraParameters(res_x=cam_2_resx, res_y=cam_2_resy, fov=cam_2_fov, R=R2, T=T2,
                                    z_near=cam_1_znear, z_far=cam_1_zfar)

    masks = get_mesh_segmentation_batch(mesh_path=mesh1_path, scale=scale,
                                        cameras_parameters=[cam_params_1, cam_params_2], device='cpu')


    cam_1_sim = CameraSimulator(resolution=(cam_1_resx, cam_1_resy), fovy=cam_1_fov, launch_viewer=False,
                                world_file="./data/world_mug.xml")
    cam_2_sim = CameraSimulator(resolution=(cam_2_resx, cam_2_resy), fovy=cam_2_fov, launch_viewer=False,
                                world_file="./data/world_mug.xml")
    cam_1_im = cam_1_sim.render(rotation_matrix=cam_1_frame_rotation, position=cam_1_pos)
    cam_2_im = cam_2_sim.render(rotation_matrix=cam_2_frame_rotation, position=cam_2_pos)

    plot_segmentation_mask(cam_1_im, masks[0])
    plot_segmentation_mask(cam_2_im, masks[1])


def test_1_camera_2_poses():
    # third example with object pose change:
    cam_3_frame_rotation = xy_axes_to_frame_rotation(cam_3_xy_axes[0], cam_3_xy_axes[1])
    R3, T3 = get_torch3d_R_T(cam_3_frame_rotation, cam_3_pos)
    cam_params_3 = CameraParameters(res_x=cam_3_resx, res_y=cam_3_resy, fov=cam_3_fov, R=R3, T=T3,
                                    z_near=cam_3_znear, z_far=cam_3_zfar)

    cam_3_sim = CameraSimulator(resolution=(cam_3_resx, cam_3_resy), fovy=cam_3_fov, launch_viewer=False,
                                world_file="./data/world_mug.xml")

    cam_3_sim.set_manipulated_object_position(obj_position_1)
    cam_3_sim.set_manipulated_object_orientation_euler(obj_orientation_1)
    cam_3_im_pose1 = cam_3_sim.render(rotation_matrix=cam_3_frame_rotation, position=cam_3_pos)

    mask_pose1 = get_mesh_segmentation_batch(mesh_path=mesh1_path, scale=scale, cameras_parameters=cam_params_3,
                                             position=obj_position_1, orientation=obj_orientation_1, device='cpu')

    plot_segmentation_mask(cam_3_im_pose1, mask_pose1[0])

    cam_3_sim.set_manipulated_object_position(obj_position_2)
    cam_3_sim.set_manipulated_object_orientation_euler(obj_orientation_2)
    cam_3_im_pose2 = cam_3_sim.render(rotation_matrix=cam_3_frame_rotation, position=cam_3_pos)

    mask_pose2 = get_mesh_segmentation_batch(mesh_path=mesh1_path, scale=scale, cameras_parameters=cam_params_3,
                                             position=obj_position_2, orientation=obj_orientation_2, device='cpu')

    plot_segmentation_mask(cam_3_im_pose2, mask_pose2[0])

if __name__ == "__main__":
    test_2_cameras_base_pose()
    test_1_camera_2_poses()


