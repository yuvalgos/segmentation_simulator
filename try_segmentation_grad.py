import torch
import numpy as np
from camera_simulator import CameraSimulator
from camera_utils import xy_axes_to_frame_rotation, get_torch3d_R_T
from geometric_mesh_segmentation import CameraParameters, get_mesh_segmentation_batch
from matplotlib import pyplot as plt
from torch import nn
from utils import masks_intersection_batch
from torchviz import make_dot


mesh = "./data/meshes/mug.obj"
scale = 0.02

cam_pos = [0.5, -0.5, 1.75]
cam_xy_axes = [[0.685, 0.728, 0.000], [-0.487, 0.458, 0.743]]
cam_frame_R = xy_axes_to_frame_rotation(cam_xy_axes[0], cam_xy_axes[1])
cam_resx, cam_resy = 300, 300
cam_fov = 45
cam_znear, cam_zfar = 0.1, 100

table_height = 1
obj_position_actual = [0, 0, 0.08 + table_height]
obj_orientation_actual = [2.1, 0, 1.57]
obj_positions = [
    [0, 0, 0.08 + table_height],
    [0, 0, 0.08 + table_height],
    [0, 0, 0.12 + table_height],
    [0, 0, 0.12 + table_height],
]
obj_orientations = [
    [2.1, 0, 1.57],
    [0, 0, 1.57],
    [0, -0.2, 0],
    [3.14, -0.2, 0]
]


class PoseSegmentation(nn.Module):
    def __init__(self, mesh_path, scale, cam_params, ref_mask, device='auto'):
        super(PoseSegmentation, self).__init__()
        self.mesh_path = mesh_path
        self.scale = scale
        self.cam_params = cam_params
        self.device = device

        self.register_buffer('ref_mask', ref_mask)

        # self.position = nn.Parameter(torch.Tensor([0, 0, 0.08 + table_height]))
        self.position = torch.Tensor([0, 0, 0.08 + table_height])
        self.position.requires_grad = False
        self.orientation = nn.Parameter(torch.Tensor([0, 0, 1.57]))
        # initial predicted pose

        self.criterion = torch.nn.MSELoss()

    def forward(self):
        mask = get_mesh_segmentation_batch(mesh_path=self.mesh_path, scale=self.scale,
                                           cameras_parameters=self.cam_params, position=self.position,
                                           orientation=self.orientation, device=self.device)
        mask = mask[:, 0, :, :]
        loss = self.criterion(mask, self.ref_mask)

        # make_dot(loss, params=dict(self.named_parameters())).view()

        return mask, loss


if __name__ == "__main__":
    position = torch.Tensor(obj_position_actual)
    orientation = torch.Tensor(obj_orientation_actual)

    R, T = get_torch3d_R_T(cam_frame_R, cam_pos)
    cam_params = CameraParameters(res_x=cam_resx, res_y=cam_resy, fov=cam_fov, R=R, T=T,
                                  z_near=cam_znear, z_far=cam_zfar)

    cam_sim = CameraSimulator(resolution=(cam_resy, cam_resx), fovy=cam_fov, )
    # load sam mask:
    sam_mask = np.load("./sam_mask.npy")
    sam_mask = torch.Tensor(sam_mask).unsqueeze(0)

    # sam_mask.requires_grad = True
    # cam_params.R.requires_grad=True

    model = PoseSegmentation(mesh_path=mesh, scale=scale, cam_params=cam_params, ref_mask=sam_mask, device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)

    for i in range(10):
        optimizer.zero_grad()

        pred_mask, loss = model()

        print("loss: ", loss.item())
        mask_intersection = masks_intersection_batch(pred_mask, sam_mask)
        plt.imshow(mask_intersection[0])
        plt.show()

        loss.backward()
        optimizer.step()


        # TODO NEXT STEPS:
        # 1. plot intersections requires additional plot util
        # 2. look at effect of blurr radius and faces per pixel, and also at learning rate

    # https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering

