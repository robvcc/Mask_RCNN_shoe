import os
import sys
import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import samples.coco.coco as coco
import matplotlib.image as mp
class InferenceConfig(coco.CocoConfig):     # --------------------设置config，增加GPU option
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
def image2mask(IMG_PATH):
	# Root directory of the project --------------------绝对路径
	ROOT_DIR = os.path.abspath("")
	sys.path.append(ROOT_DIR)  # To find local version of the library -------------增加根目录

	sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs") # --------------------保存logs和训练好的模型

	# Local path to trained weights file   # --------------------  权重文件

	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h6")
	# Download COCO trained weights from Releases if needed#  --------------------下载权重文件
	# print(COCO_MODEL_PATH)
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")# --------------------图片路径

	# config.display()

	# Create model object in inference mode. --------------------模型加载
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO --------------------权重加载
	# print(COCO_MODEL_PATH) #权重路径
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	# COCO Class names
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
				   'bus', 'train', 'truck', 'boat', 'traffic light',
				   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
				   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
				   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
				   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
				   'kite', 'baseball bat', 'baseball glove', 'skateboard',
				   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
				   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
				   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
				   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
				   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
				   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
				   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
				   'teddy bear', 'hair drier', 'toothbrush']

	# Load a random image from the images folder
	# print(IMAGE_DIR)
	# file_names = next(os.walk(IMAGE_DIR))[2]
	# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
	# image = skimage.io.imread(os.path.join(IMAGE_DIR, "12283150_12d37e6389_z.jpg")) # ------------- 相对路径选择图片
	

	# Run detection
	image = skimage.io.imread(IMG_PATH)# --- 绝对路径选择图片
	results = model.detect([image], verbose=1)# -------------预测
	# Visualize results
	r = results[0]
	# print(r)
	img=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])# -------------可视化
    # mp.imsave("C:/Users/VCC/Desktop/Image_After_MaskRCNN.jpg",img)

#

IMG_PATH = "C:/Users/VCC/Desktop/quilt and baby/rgb_24.jpg"
# IMG_PATH = "C:/Users/VCC/Desktop/pic/4.png"

image2mask(IMG_PATH)

