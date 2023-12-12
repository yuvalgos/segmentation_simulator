import numpy as np
import torch
from camera_simulator import CameraSimulator
from camera_utils import xy_axes_to_frame_rotation, get_torch3d_R_T
from matplotlib import pyplot as plt
from sam_segmentation import SAMSegmentation
from utils import plot_segmentation_mask, plot_grid_segmentation_masks, masks_intersection_batch,\
    compute_masks_IOU_batch
from geometric_mesh_segmentation import CameraParameters, get_mesh_segmentation_batch

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
# TODO move to stable pose set file.
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

if __name__ == "__main__":
    mask = np.zeros([cam_resy, cam_resx])

    cam_sim = CameraSimulator(resolution=(cam_resy, cam_resx), fovy=cam_fov, )

    # generate 4 images for four poses and plot them side by side:
    images = []
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for obj_position, obj_orientation, ax in zip(obj_positions, obj_orientations, axs):
        cam_sim.set_manipulated_object_position(obj_position)
        cam_sim.set_manipulated_object_orientation_euler(obj_orientation)
        im = cam_sim.render(cam_frame_R, cam_pos)
        ax.imshow(im)
        ax.axis('off')
        images.append(im)
    plt.show()

    # plot four geometric segmentations (predicted segmentation):
    masks = []
    for obj_position, obj_orientation in zip(obj_positions, obj_orientations):
        # can't really batch here since pose is different
        R, T = get_torch3d_R_T(cam_frame_R, cam_pos)
        cam_params = CameraParameters(res_x=cam_resx, res_y=cam_resy, fov=cam_fov, R=R, T=T,
                                      z_near=cam_znear, z_far=cam_zfar)
        mask = get_mesh_segmentation_batch(mesh_path=mesh, scale=scale, cameras_parameters=cam_params,
                                           position=obj_position, orientation=obj_orientation, device='cpu')
        masks.append(mask)
    plot_grid_segmentation_masks(images, masks, (1, 4), mask_alpha=0.95)

    # use sam to segment actual image and show it
    cam_sim.set_manipulated_object_position(obj_position_actual)
    cam_sim.set_manipulated_object_orientation_euler(obj_orientation_actual)
    im_actual = cam_sim.render(cam_frame_R, cam_pos)

    sam = SAMSegmentation()
    mask_sam, score = sam.segment_image_center(im_actual, best_out_of_3=True)
    plot_segmentation_mask(im_actual, mask_sam, mask_alpha=0.95, color=[30, 255, 30])
    iou = compute_masks_IOU_batch(torch.from_numpy(mask_sam).unsqueeze(0), torch.cat(masks))

    # plot intersection images:
    masks_geometric = torch.cat(masks, dim=0)
    masks_sam = torch.from_numpy(mask_sam).unsqueeze(0)
    intersection = masks_intersection_batch(masks_geometric, masks_sam)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for im, ax, iou_score in zip(intersection, axs, iou):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(f"IOU: {iou_score.item():.2f}")
        ax.title.set_size(30)
    plt.show()

    # use softmax to get pose distribution, with high low temperature:
    pose_distribution = torch.softmax(iou * 3, dim=0).numpy().squeeze()
    # plot pose distribution wide:
    plt.figure(figsize=(20, 5))
    plt.bar(range(len(pose_distribution)), pose_distribution)
    plt.xticks(range(len(pose_distribution)), ["Pose 1", "Pose 2", "Pose 3", "Pose 4"])
    plt.title("Pose distribution", fontsize=30)
    plt.xticks(fontsize=20)
    plt.show()

