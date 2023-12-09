from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np


class SAMSegmentation:
    def __init__(self, device='auto'):
        sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
        self.predictor = SamPredictor(sam)

    def segment_image_center(self, im, best_out_of_3=True):
        h, w = im.shape[:2]
        point_coords = np.array([[w/2, h/2]])
        point_labels = np.array([1])

        self.predictor.set_image(im)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=best_out_of_3,
        )

        if not best_out_of_3:
            return masks[0], scores[0]
        else:
            best_mask = np.argmax(scores)
            return masks[best_mask], scores[best_mask]

        # TODO: create a file to test this. remove above code from this file, it's still in segment.py
        # TODO: the test file should run simulation and be base for example, call it example.py

        # TODO: batched images in the end: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
