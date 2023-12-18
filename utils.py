import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_segmentation_mask(im, mask, mask_alpha=0.8, color=np.array([255, 30, 30])):
    colored_mask = im.copy()
    colored_mask[mask.squeeze() != 0] = color
    segmented_im = (colored_mask/255) * mask_alpha + (im / 255) * (1 - mask_alpha)
    plt.imshow(segmented_im)
    plt.show()


def plot_grid_segmentation_masks(images, masks, row_cols=(1,4), mask_alpha=0.8, color=np.array([255, 30, 30])):
    fig, axs = plt.subplots(row_cols[0], row_cols[1], figsize=(20, 5))
    axs = axs.reshape(-1)
    for im, mask, ax in zip(images, masks, axs):
        colored_mask = im.copy()
        colored_mask[mask.squeeze() != 0] = color
        segmented_im = (colored_mask/255) * mask_alpha + (im / 255) * (1 - mask_alpha)
        ax.imshow(segmented_im)
        ax.axis('off')
    plt.show()


def masks_intersection_batch(mask1, mask2, mask1_color=np.array([255, 30, 30]), mask2_color=np.array([30, 255, 30])):
    """
    create image of intersection between two masks, mask1 and mask2 should be batches of the same size NXHXW or one of
    them should be a batch and the other a single mask.
    :param mask1:
    :param mask2:
    :param mask1_color:
    :return: batch of masks intersection images
    """
    # make sure masks are binary:
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # add color dimension and repeat mask in that dimention:
    mask_1_colored = mask1.unsqueeze(1).repeat(1, 3, 1, 1)
    mask_2_colored = mask2.unsqueeze(1).repeat(1, 3, 1, 1)

    mask_1_colored = mask_1_colored * torch.Tensor(mask1_color).reshape(1, 3, 1, 1)
    mask_2_colored = mask_2_colored * torch.Tensor(mask2_color).reshape(1, 3, 1, 1)

    intersection = mask_1_colored + mask_2_colored
    intersection[intersection > 255] = 255

    intersection = intersection.permute(0, 2, 3, 1)
    intersection = intersection / 255

    return intersection

def compute_masks_IOU_batch(mask1, mask2):
    """
    compute IOU between two masks, mask1 and mask2 should be batches of the same size NXHXW or one of
    them should be a batch and the other a single mask.
    :param mask1:
    :param mask2:
    :return: batch of IOU values
    """
    # make sure masks are binary:
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = mask1 * mask2
    union = mask1 + mask2
    union[union > 1] = 1

    intersection = intersection.reshape(intersection.shape[0], -1).sum(dim=1)
    union = union.reshape(union.shape[0], -1).sum(dim=1)

    return intersection / union
