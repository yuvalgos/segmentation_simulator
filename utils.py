import numpy as np
import matplotlib.pyplot as plt


def plot_segmentation_mask(im, mask, mask_alpha=0.8, color=np.array([255, 30, 30])):
    colored_mask = im.copy()
    colored_mask[mask.squeeze() != 0] = color
    segmented_im = (colored_mask/255) * mask_alpha + (im / 255) * (1 - mask_alpha)
    plt.imshow(segmented_im)
    plt.show()

