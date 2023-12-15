import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.renderer.mesh import Textures, MeshRenderer, MeshRasterizer, SoftSilhouetteShader
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.transforms import euler_angles_to_matrix
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List, Union


@dataclass
class CameraParameters:
    """convinience package of camera parameters"""

    res_x: int = 200
    res_y: int = 200
    fov: float = 60
    R: torch.Tensor = torch.eye(3)
    T: torch.Tensor = torch.zeros(3)
    z_near: float = 0.01
    z_far: float = 1000

    # aspect ratio property from res:
    @property
    def aspect_ratio(self):
        return 1
        # TODO remove?

    @staticmethod
    def batch_camera_parameters(cameras_parameters: List['CameraParameters']):
        """
        :param cameras_parameters: list of CameraParameters objects
        :return: batch for each parameter
        """

        res_x = torch.tensor([cam.res_x for cam in cameras_parameters])
        res_y = torch.tensor([cam.res_y for cam in cameras_parameters])
        fov = torch.tensor([cam.fov for cam in cameras_parameters])
        R = torch.stack([cam.R for cam in cameras_parameters])
        T = torch.stack([cam.T for cam in cameras_parameters])
        z_near = torch.tensor([cam.z_near for cam in cameras_parameters])
        z_far = torch.tensor([cam.z_far for cam in cameras_parameters])
        aspect_ratio = torch.tensor([cam.aspect_ratio for cam in cameras_parameters])

        return res_x, res_y, fov, R, T, z_near, z_far, aspect_ratio


def transform_mesh(verts, position, orientation):
    """
    transform mesh by position and orientation
    :param verts: mesh vertices
    :param position: x, y, z position of the object in the world
    :param orientation: euler orientation of the object in the world, as x, y, z angles
    :return: transformed mesh vertices
    """
    # account for the different coordinate systems between torch3d and mujoco. The angles are used in mujoco normally.
    euler_new = torch.Tensor([orientation[2], -orientation[1], -orientation[0]])

    # R = Rotation.from_euler('xyz', euler_new, degrees=False).as_matrix()
    # R = torch.Tensor(R)
    # equivalent but only with torch:
    R = euler_angles_to_matrix(torch.flip(euler_new, [0]), "ZYX")

    verts_transofrmed = R @ verts.T

    pos_torch = torch.Tensor([-position[0], -position[1], position[2]])
    return verts_transofrmed.T + pos_torch


def load_mesh(mesh_paths, scale, position, oreintation, device) -> Meshes:
    """
    load mesh from file to torch mesh object.
    """
    verts, faces, aux = load_obj(mesh_paths)
    # scale the mesh:
    verts = verts * scale
    verts = transform_mesh(verts, position, oreintation)
    verts_rgb = torch.ones(verts.shape[0], 3, dtype=torch.float32).unsqueeze(0).to(device)
    mesh = Meshes(
        verts=verts.unsqueeze(0).to(device),
        faces=faces.verts_idx.unsqueeze(0).to(device),
        textures=TexturesVertex(verts_features=verts_rgb)
    )

    return mesh


def get_cameras(cameras_parameters: Union[List[CameraParameters], CameraParameters], device='auto'):
    """
    :param cameras_parameters: CameraParameters object list or a  single CameraParameters object
    :param device: torch device
    :return: batch of cameras
    """
    if isinstance(cameras_parameters, CameraParameters):
        cameras_parameters = [cameras_parameters]

    res_x, res_y, fov, R, T, z_near, z_far, aspect_ratio = CameraParameters.batch_camera_parameters(cameras_parameters)

    cameras = FoVPerspectiveCameras(znear=z_near, zfar=z_far, aspect_ratio=aspect_ratio, fov=fov, R=R, T=T,
                                    device=device)
    return cameras, res_x, res_y


def get_mesh_segmentation_batch(mesh_path, cameras_parameters: Union[List[CameraParameters], CameraParameters], scale=1,
                                position=[0,0,0], orientation=[0,0,0], device='auto'):
    """

    :param mesh_path: path to the mesh, one mesh per batch. same for position, orientation and scale
    :param cameras_parameters: list of camera parameters objects for different cameras. can pass only one camera.
    :param scale: mesh scale, applied to the vertices
    :param position: x, y, z position of the object in the world
    :param orientation: euler orientation of the object in the world, as x, y, z angles
    :param device: torch evice
    :return: batch of segmentation mask_pose1
    """

    n_cameras = 1 if isinstance(cameras_parameters, CameraParameters) else len(cameras_parameters)

    mesh = load_mesh(mesh_path, scale, position, orientation, device)
    cameras, res_x, res_y = get_cameras(cameras_parameters, device)
    if n_cameras != 1:
        mesh = mesh.extend(n_cameras)

    # right now, now support for different resolution in batch
    assert torch.all(res_x == res_x[0]), "all images in batch must have the same resolution"
    assert torch.all(res_y == res_y[0]), "all images in batch must have the same resolution"

    raster_settings = RasterizationSettings(image_size=(int(res_x[0]), int(res_y[0])), blur_radius=0.0,
                                            faces_per_pixel=1)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=SoftPhongShader(device=device, cameras=cameras))

    images = renderer(mesh)
    masks = (images[..., 3]).unsqueeze(1)

    return masks