from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np


class SAMSegmentation:
    def __init__(self, device='auto'):
        self.download_weights_if_needed()
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

    def download_weights_if_needed(self):
        import os
        import urllib.request
        if os.path.exists("./models/sam_vit_b_01ec64.pth"):
            return

        print("Downloading SAM weights...")
        # make models dir if needed:
        os.makedirs("./models", exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        # download, original file name but to models dir:
        urllib.request.urlretrieve(url, "./models/sam_vit_b_01ec64.pth")
        print("Done.")