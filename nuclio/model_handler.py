import os
import cv2
import numpy as np
import json
import base64
import io
from PIL import Image
import sys
sys.path.insert(0, "Mask2Former")

from skimage.measure import approximate_polygon, find_contours

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES

from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config


class ModelHandler:
    def __init__(self, labels):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        print(os.getcwd())
        cfg.merge_from_file("Mask2Former/configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml") 
        cfg.MODEL.WEIGHTS = "model_final_4ab90c.pkl"
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        predictor = DefaultPredictor(cfg)       
        self.model = predictor
        self.labels = labels

    def infer(self, image): 
        results = []
        output = self.model(image)
        
        masks = output["sem_seg"]
        print(masks.shape)
        height, width = masks[1,:,:].shape
        print(width, height)

        for i in range(len(masks[:,0,0])):
            
            mask_by_label = masks[i,:,:].cpu().numpy()
            mask_by_label = cv2.resize(mask_by_label, dsize=(width, height),interpolation=cv2.INTER_CUBIC)  #dsize=(image.width, image.height)

            contours = find_contours(mask_by_label, 0.8)

            for contour in contours:
                contour = np.flip(contour, axis=1)
                contour = approximate_polygon(contour, tolerance=2.5)
                if len(contour) < 3:
                    continue

                results.append({
                    "confidence": None,
                    "label": self.labels.get(i, "unknown"),
                    "points": contour.ravel().tolist(),
                    "type": "polygon",
                })

        return results

