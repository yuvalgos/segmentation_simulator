from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


im = cv2.imread("mug.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image(im)

plt.imshow(im)

points_fg = None
points_labels = None
points_fg = np.array([[110,110]])
points_labels = np.array([1])
# show_points(points_fg, points_labels, plt.gca())
# plt.show()

box = None
# box = np.array([[100, 75, 190, 160]])
# show_box(box[0], plt.gca())

# segment:
masks, scores, logits = predictor.predict(
    point_coords=points_fg,
    point_labels=points_labels,
    box=box,
    multimask_output=True,
)

# plot image and mask_pose1:
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    show_mask(mask, plt.gca())
    show_points(points_fg, points_labels, plt.gca())
    # show_box(box[0], plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

# plot all mask_pose1:
# for mask in mask_pose1:
#     m = mask['segmentation']
#     plt.imshow(m)
#     plt.show()
