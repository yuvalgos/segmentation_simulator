import numpy as np
from camera_simulator import CameraSimulator
import matplotlib.pyplot as plt
import time
from camera_utils import homogeneous_to_cartesian, cartesian_to_homogeneous, intrinsic_matrix_from_params, \
    extrinsic_matrix_from_rotation_translatin, xy_axes_to_frame_rotation
from configurations import res_x, res_y, fov_x, fov_y, cam_1_pos, cam_1_xy_axes, pole_interest_points
# from meshlib import mrmeshpy
import pymeshlab
from scipy.spatial.transform import Rotation

import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.renderer.mesh import Textures, MeshRenderer, MeshRasterizer, SoftSilhouetteShader
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer.mesh.textures import TexturesVertex
from torch.nn import functional as F


file_name = './data/meshes/mug.obj'
scale =  0.05

def render_mask(img_size, R, T, fov_y, aspect_ratio=1.0, device='auto'):
    # need to look at it to support fov_x with aspect ratio
    # will render a batch of cameras

    # TODO: this is for the case where there is one mesh? change to batch
    verts, faces, aux = load_obj(file_name)
    # scale the mesh:
    verts = verts * scale
    verts_rgb = torch.ones(verts.shape[0], 3, dtype=torch.float32).unsqueeze(0).to(device)
    mesh = Meshes(
        verts=verts.unsqueeze(0).to(device),
        faces=faces.verts_idx.unsqueeze(0).to(device),
        textures=TexturesVertex(verts_features=verts_rgb)
    )

    focal_length_tensor = torch.tensor([fx, fy], dtype=torch.float32).unsqueeze(0)
    cameras = FoVPerspectiveCameras(znear=0.01, zfar=1000, aspect_ratio=aspect_ratio, fov=fov_y,
                                    device=device, R=R, T=T)
    # cameras = PerspectiveCameras(focal_length=focal_length_tensor, R=R, T=T, device=device,
    #                              image_size=img_size, in_ndc=False) # True is default for in_ndc
    im_size_int_tuple = (int(img_size[0,0]), int(img_size[0,1]))
    raster_settings = RasterizationSettings(image_size=im_size_int_tuple, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )

    images = renderer(mesh) # B x 480 x 640 x 4
    masks = (images[...,3]).unsqueeze(1) # B x 1 x 480 x 640
    im = images[..., :3]
    # mask_pose1 = F.interpolate(mask_pose1, scale_factor=(0.5)) # downsample to B x 1 x 240 x 320

    return masks


def get_R_T(cam_frame_rotation, cam_pos):
    x, y, z = cam_frame_rotation[:, 0], cam_frame_rotation[:, 1], cam_frame_rotation[:, 2]
    cam_R = np.concatenate((x.reshape(3, 1), y.reshape(3, 1), z.reshape(3, 1)), axis=1)

    muj2torch = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    R = muj2torch @ cam_R.T

    T = - R @ cam_pos
    # cam pos: 2, -2, 3

    R = torch.tensor(R, dtype=torch.float32).unsqueeze(0)
    T = torch.tensor(T, dtype=torch.float32).unsqueeze(0)

    return R, T

if __name__ == '__main__':
    res_x, res_y = 200, 200
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    intrinsic_matrix = intrinsic_matrix_from_params(res_x, res_y, fov_x, fov_y)

    # set of parameters for extrinsics, camera at the corner looking down to the table:
    cam_pos = cam_1_pos
    cam_frame_rotation = xy_axes_to_frame_rotation(cam_1_xy_axes[0], cam_1_xy_axes[1])

    # cam_frame_rotation = np.eye(3)
    # cam_frame_rotation = np.array([[0.7071068, -0.7071068, 0.000], [0.7071068, 0.7071068, 0], [0, 0, 1]]).T # 45deg around Z works
    # cam_frame_rotation = np.array([[0.9396926, 0, 0.3420202], [0, 1, 0], [-0.3420202, 0, 0.9396926]]).T # 20deg around Y
    # cam_frame_rotation = np.array([[1,  0,  0], [0,  0.9396926, -0.3420202], [0,  0.3420202,  0.9396926]]) # 20deg around X
    # cam_pos = np.array([-0.3,0.4,2])

    # extrinsic_matrix = extrinsic_matrix_from_rotation_translatin(cam_frame_rotation, cam_pos)

    # camera_matrix = intrinsic_matrix @ extrinsic_matrix

    R, T = get_R_T(cam_frame_rotation, cam_pos)
    R = R.to(device)
    T = T.to(device)

    fx = torch.tensor(intrinsic_matrix[0, 0], dtype=torch.float32).unsqueeze(0).to(device)
    fy = torch.tensor(intrinsic_matrix[1, 1], dtype=torch.float32).unsqueeze(0).to(device)
    img_size = torch.tensor([res_x, res_y], dtype=torch.int32).unsqueeze(0).to(device)
    mask = render_mask(img_size, R, T, fov_y=fov_y, device=device)

    # plot mask:
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.show()

    # TODO: use 4x4 matrix for extrinsics to check if target is at FOV
    cam_sim = CameraSimulator(resolution=(res_x, res_y), fovy=fov_y ,launch_viewer=False,
                              world_file="./data/world_mug.xml")
    res = cam_sim.render(rotation_matrix=cam_frame_rotation, position=cam_pos)
    # time.sleep(60)

    plt.imshow(res)
    plt.show()

    # save res:
    plt.imsave('mug.png', res)

    # plt.plot(interest_points_im_plane[:, 0], interest_points_im_plane[:, 1], '.', color='red',
    #          markersize=1, alpha=0.5)
    # plt.show()

    time.sleep(1)

