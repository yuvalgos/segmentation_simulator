import numpy as np
import torch
from camera_simulator import CameraSimulator
from camera_utils import xy_axes_to_frame_rotation, get_torch3d_R_T
from matplotlib import pyplot as plt
from sam_segmentation import SAMSegmentation
from utils import plot_segmentation_mask, plot_grid_segmentation_masks, masks_intersection_batch, \
    compute_masks_IOU_batch, plot_grid_masked_depth_on_image, compute_depth_squared_distance_in_intersection, \
    masked_depth_diff
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
actual_pose = 0
obj_position_actual = obj_positions[actual_pose]
obj_orientation_actual = obj_orientations[actual_pose]

max_depth = 1.5

if __name__ == "__main__":

    cam_sim = CameraSimulator(resolution=(cam_resy, cam_resx), fovy=cam_fov, )

    # generate 4 images and 4 depth maps for four poses and plot them side by side:
    images, depth_images = [], []
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for obj_position, obj_orientation, ax_im, ax_depth in zip(obj_positions, obj_orientations, axs[0, :], axs[1, :]):
        cam_sim.set_manipulated_object_position(obj_position)
        cam_sim.set_manipulated_object_orientation_euler(obj_orientation)
        im = cam_sim.render(cam_frame_R, cam_pos)
        im_depth = cam_sim.render_depth(cam_frame_R, cam_pos)
        im_depth[im_depth > max_depth] = -1

        ax_im.imshow(im)
        ax_depth.imshow(im_depth)
        ax_im.axis('off')
        ax_depth.axis('off')

        images.append(im)
        depth_images.append(im_depth)
    plt.show()

    # plot four geometric segmentations (predicted segmentation) and four depth maps of the segmented area
    # (masked predicted depth):
    masks_pred, depth_pred = [], []
    for obj_position, obj_orientation in zip(obj_positions, obj_orientations):
        obj_position = torch.Tensor(obj_position)
        obj_orientation = torch.Tensor(obj_orientation)
        # can't really batch here since pose is different
        R, T = get_torch3d_R_T(cam_frame_R, cam_pos)
        cam_params = CameraParameters(res_x=cam_resx, res_y=cam_resy, fov=cam_fov, R=R, T=T,
                                      z_near=cam_znear, z_far=cam_zfar)
        mask, depth = get_mesh_segmentation_batch(mesh_path=mesh, scale=scale, cameras_parameters=cam_params,
                                                  position=obj_position, orientation=obj_orientation, device='cpu',
                                                  return_depth_map=True)
        masks_pred.append(mask)
        depth_pred.append(depth)
    plot_grid_segmentation_masks(images, masks_pred, (1, 4), mask_alpha=0.95, title="Geometric segmentation")
    plot_grid_masked_depth_on_image(images, depth_pred, masks_pred, (1, 4), mask_alpha=0.99,
                                    title="Depth map masked by segmentation")

    # use sam to segment actual image and show it
    cam_sim.set_manipulated_object_position(obj_position_actual)
    cam_sim.set_manipulated_object_orientation_euler(obj_orientation_actual)
    im_actual = cam_sim.render(cam_frame_R, cam_pos)
    depth_actual = cam_sim.render_depth(cam_frame_R, cam_pos)

    sam = SAMSegmentation()
    mask_sam, score = sam.segment_image_center(im_actual, best_out_of_3=True)
    plot_segmentation_mask(im_actual, mask_sam, mask_alpha=0.95, color=[30, 255, 30])

    mask_sam_torch = torch.from_numpy(mask_sam).unsqueeze(0)
    depth_actual_torch = torch.from_numpy(depth_actual).unsqueeze(0)
    masks_pred_batch = torch.cat(masks_pred)
    depth_pred = torch.cat(depth_pred)

    # save depth and mask:
    torch.save(depth_actual_torch, "./ref_depth.pt")
    torch.save(mask_sam_torch, "./ref_mask.pt")

    # plot intersection images:
    iou = compute_masks_IOU_batch(mask_sam_torch, masks_pred_batch)
    depth_distance = compute_depth_squared_distance_in_intersection(depth_actual_torch, depth_pred,
                                                                    mask_sam_torch, masks_pred_batch)
    intersection = masks_intersection_batch(masks_pred_batch, mask_sam_torch)
    masked_depth_diff = masked_depth_diff(depth_pred, depth_actual_torch, masks_pred_batch, mask_sam_torch)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for intr, depth_diff, ax_intr, ax_depth, iou_score, depth_dist in \
            zip(intersection,masked_depth_diff, axs[0], axs[1], iou, depth_distance):
        ax_intr.imshow(intr)
        ax_intr.axis('off')
        ax_intr.set_title(f"IOU: {iou_score.item():.2f}")
        ax_intr.title.set_size(25)
        ax_depth.imshow(depth_diff, cmap='gray')
        ax_depth.axis('off')
        ax_depth.set_title(f"Depth diff: {depth_dist.item():.6f}")
        ax_depth.title.set_size(25)

    plt.show()


    iou = iou.squeeze()
    depth_distance = depth_distance.squeeze()
    normalized_depth_distance = depth_distance / depth_distance.max()
    pose_distribution = iou + (1 - depth_distance)
    pose_distribution = pose_distribution.numpy()
    pose_distribution = pose_distribution / pose_distribution.sum()

    # plot pose distribution wide:
    plt.figure(figsize=(20, 5))
    plt.bar(range(len(pose_distribution)), pose_distribution)
    plt.xticks(range(len(pose_distribution)), ["Pose 1", "Pose 2", "Pose 3", "Pose 4"])
    plt.title("Pose distribution", fontsize=30)
    plt.xticks(fontsize=20)
    plt.show()
