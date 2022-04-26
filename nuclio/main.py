import json
import base64
import io
from PIL import Image
import sys
sys.path.insert(0, "Mask2Former")

import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch

#install skimage
from skimage.measure import approximate_polygon, find_contours

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES


# import Mask2Former project
from mask2former import add_maskformer2_config

from model_handler import ModelHandler

def init_context(context):
    
    context.logger.info("Init context...  0%")
    
    #segmantic segmentation starts from 0 in output
    labels = {n: item['name'] for n, item in enumerate(CITYSCAPES_CATEGORIES)}
    
    #model specified in ModelHandler
    model = ModelHandler(labels)
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Mask2Former model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
    results = context.user_data.model.infer(image)

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
