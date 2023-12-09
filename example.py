import numpy as np
from camera_simulator import CameraSimulator
from camera_utils import xy_axes_to_frame_rotation
from sam_segmentation import SAMSegmentation
from utils import plot_segmentation_mask

cam_pos = [0.5, -0.5, 1.75]
cam_xy_axes = [[0.685, 0.728, 0.000], [-0.487, 0.458, 0.743]]
cam_frame_R = xy_axes_to_frame_rotation(cam_xy_axes[0], cam_xy_axes[1])
cam_resx, cam_resy = 300, 300
cam_fov = 45
cam_znear, cam_zfar = 0.1, 100

table_height = 1
obj_position = [0, 0, 0.08 + table_height]
obj_orientation = [2.1, 0, 1.57]  # TODO move to stable pose set file.
obj_orientations = [
    [2.1, 0, 1.57]
]


if __name__ == "__main__":
    mask = np.zeros([cam_resy, cam_resx])

    cam_sim = CameraSimulator(resolution=(cam_resy, cam_resx), fovy=cam_fov,)
    cam_sim.set_manipulated_object_position(obj_position)
    cam_sim.set_manipulated_object_orientation_euler(obj_orientation)
    im = cam_sim.render(cam_frame_R, cam_pos)

    # sam = SAMSegmentation()
    # mask, score = sam.segment_image_center(im, best_out_of_3=True)

    plot_segmentation_mask(im, mask)

