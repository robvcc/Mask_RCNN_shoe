import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib
from mrcnn import utils

from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shape"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 6 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
# config.display()


class Detect_Config(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
detectconfig=Detect_Config()
model = modellib.MaskRCNN(mode="inference", config=detectconfig,
                          model_dir=MODEL_DIR)
SHARP_MODEL_DIR=os.path.join(MODEL_DIR,"logs/shape20190608T1400")
#SHARP_MODEL_PATH=os.path.join(SHARP_MODEL_DIR,"mask_rcnn_shapes_0000.h5")
SHARP_MODEL_PATH="logs/shape20190608T1400/mask_rcnn_shape_0110.h5"
model.load_weights(SHARP_MODEL_PATH, by_name=True)
print(SHARP_MODEL_PATH)