from geometric_mesh_segmentation import CameraParameters, get_mesh_segmentation_batch
from camera_utils import xy_axes_to_frame_rotation, get_torch3d_R_T
import matplotlib.pyplot as plt
import numpy as np
from camera_simulator import CameraSimulator
from utils import plot_segmentation_mask


# parameters:
mesh1_path = "./data/meshes/mug.obj"

cam_1_pos = [2, -2, 3]
cam_1_xy_axes = [[0.685, 0.728, 0.000], [-0.487, 0.458, 0.743]]
cam_1_resx, cam_1_resy = 200, 300
cam_1_fov = 60
cam_1_znear, cam_1_zfar = 0.1, 100

cam_2_pos = np.array([-0.3,0.4,2])
cam_2_frame_rotation = np.array([[0.7071068, -0.7071068, 0.000], [0.7071068, 0.7071068, 0], [0, 0, 1]]).T # 45deg around Z
cam_2_resx, cam_2_resy = 200,300
cam_2_fov = 45
cam_2_znear, cam_2_zfar = cam_1_znear, cam_1_zfar

if __name__ == "__main__":
    cam_1_frame_rotation = xy_axes_to_frame_rotation(cam_1_xy_axes[0], cam_1_xy_axes[1])

    R1, T1 = get_torch3d_R_T(cam_1_frame_rotation, cam_1_pos)
    cam_params_1 = CameraParameters(res_x=cam_1_resx, res_y=cam_1_resy, fov=cam_1_fov, R=R1, T=T1,
                                    z_near=cam_1_znear, z_far=cam_1_zfar)


    R2, T2 = get_torch3d_R_T(cam_2_frame_rotation, cam_2_pos)
    cam_params_2 = CameraParameters(res_x=cam_2_resx, res_y=cam_2_resy, fov=cam_2_fov, R=R2, T=T2,
                                    z_near=cam_1_znear, z_far=cam_1_zfar)

    masks = get_mesh_segmentation_batch(mesh_path=mesh1_path, scale=0.05,
                                        cameras_parameters=[cam_params_1, cam_params_2], device='cpu')


    cam_1_sim = CameraSimulator(resolution=(cam_1_resx, cam_1_resy), fovy=cam_1_fov, launch_viewer=False,
                                world_file="./data/world_mug.xml")
    cam_2_sim = CameraSimulator(resolution=(cam_2_resx, cam_2_resy), fovy=cam_2_fov, launch_viewer=False,
                                world_file="./data/world_mug.xml")
    cam_1_im = cam_1_sim.render(rotation_matrix=cam_1_frame_rotation, position=cam_1_pos)
    cam_2_im = cam_2_sim.render(rotation_matrix=cam_2_frame_rotation, position=cam_2_pos)

    plot_segmentation_mask(cam_1_im, masks[0])
    plot_segmentation_mask(cam_2_im, masks[1])
