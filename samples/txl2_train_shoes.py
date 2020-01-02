#-*-coding:utf-8-*-

import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 8 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            # temp = yaml.load(f.read())
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,可通过这种方式扩展多个物体
        self.add_class("shapes", 1, "shoe_l")
        self.add_class("shapes", 2, "shoe_r")
        self.add_class("shapes", 3, "mouth")
        self.add_class("shapes", 4, "rack")
        # self.add_class("shapes", 5, "shoe_rack")


        for i in range(count):
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            #filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            # print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "pic/" + filestr + ".png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("shoe_l") != -1:
                labels_form.append("shoe_l")
            elif labels[i].find("shoe_r") != -1:
                labels_form.append("shoe_r")
            elif labels[i].find("mouth") != -1:
                labels_form.append("mouth")
            elif labels[i].find("rack") != -1:
                labels_form.append("rack")
            # elif labels[i].find("shoe_rack") != -1:
            #     labels_form.append("shoe_rack")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


class Detect_Config(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1




#shoes_train_class
class Shoes():
    def __init__(self):
        self.ROOT_DIR = os.path.abspath("..")
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        print(self.MODEL_DIR)
        self.iter_num = 0
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        # self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "logs/shape20191125T1931/mask_rcnn_shape_0012.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        self.config = ShapesConfig()
        # self.config.display()

        self.dataset_root_path=self.ROOT_DIR+"/txl_shoes_data2/"
        #self.dataset_root_path="/home/txl/PycharmProjects/Mask_RCNN_shoes/shoes_data/"
        self.img_floder = self.dataset_root_path + "pic"
        # print(img_floder)
        self.mask_floder = self.dataset_root_path + "cv2_mask"
        #yaml_floder = dataset_root_path
        self.imglist = os.listdir(self.img_floder)
        self.count = len(self.imglist)
        #print(self.imglist)
        #print(self.count)

        self.val_root_path = self.ROOT_DIR+"/val_data/"
        self.val_img_floder = self.val_root_path + "pic"
        self.val_mask_floder = self.val_root_path + "cv2_mask"
        self.val_imglist = os.listdir(self.val_img_floder)
        self.val_count = len(self.val_imglist)

        self.detectconfig = Detect_Config()

    def prepare_data(self):
        #train与val数据集准备
        self.dataset_train = DrugDataset()
        self.dataset_train.load_shapes(self.count, self.img_floder, self.mask_floder, self.imglist, self.dataset_root_path)
        self.dataset_train.prepare()
        self.dataset_val = DrugDataset()
        self.dataset_val.load_shapes(self.val_count, self.val_img_floder, self.val_mask_floder, self.val_imglist, self.val_root_path)
        # self.dataset_val.load_shapes(10, self.img_floder, self.mask_floder, self.imglist, self.dataset_root_path)
        self.dataset_val.prepare()
        print("dataset_val-->",self.dataset_val._image_ids)

    def load_pretrain_model(self):
        self.model = modellib.MaskRCNN(mode="training", config=self.config,
                                  model_dir=self.MODEL_DIR)
        init_with = "coco"  # imagenet, coco, or last
        if init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            self.model.load_weights(self.COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            self.model.load_weights(self.model.find_last()[1], by_name=True)

    def tarin(self):
        self.model.train(self.dataset_train, self.dataset_val,
                    learning_rate=self.config.LEARNING_RATE/10,
                    epochs=110,
                    layers='all')

    def detect(self):
        self.model = modellib.MaskRCNN(mode="inference", config=self.detectconfig,
                                  model_dir=self.MODEL_DIR)
        # SHARP_MODEL_PATH=os.path.join(SHARP_MODEL_DIR,"mask_rcnn_shapes_0000.h5")
        # self.SHARP_MODEL_PATH="/home/ljt/Mask_RCNN_shoes/logs/shape20190613T1601/mask_rcnn_shape_0010.h5"
        #self.SHARP_MODEL_PATH="../logs/shape20190613T1601/mask_rcnn_shape_0010.h5"
        self.SHARP_MODEL_PATH="../logs/shape20190903T1841/mask_rcnn_shape_0010.h5"
        self.model.load_weights(self.SHARP_MODEL_PATH, by_name=True)
        print(self.SHARP_MODEL_PATH)

        import skimage
        # Quilt_DIR="/home/ljt/Shoe-data-V2/test"
        Quilt_DIR="../shoes_data/test"

        IMAGE_DIR=os.path.join(Quilt_DIR,"/")
        #image = skimage.io.imread(os.path.join(IMAGE_DIR, "17.png"))
        #image = skimage.io.imread("/home/ljt/Shoe-data-V2/test/1.png")
        # image = skimage.io.imread("C:/Users/VCC/Desktop/3.jpg")
        # Run detection
        # print(image.shape)
        #image = cv2.imread(os.path.join(IMAGE_DIR, "65.png"))
        # image = cv2.imread("/home/ljt/Shoe-data-V2/test/20.png")
        # image = cv2.imread("../shoes_data/test/578.jpg")
        image = cv2.imread("../shoes_data/test/shoes.jpg")
        results = self.model.detect([image], verbose=1)
        r = results[0]
        a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        self.dataset_val.class_names, r['scores'])

    def mAP(self):
        image_ids = np.random.choice(self.dataset_val.image_ids, 10)
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(self.dataset_val, self.detectconfig,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.detectconfig), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        print("mAP: ", np.mean(APs))

    def acc(self):
        image_ids = np.random.choice(self.dataset_val.image_ids, 20)
        acc = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(self.dataset_val, self.detectconfig,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.detectconfig), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # if not r["scores"]:
            #     r["scores"] = [0.5662385]
            if not any(r["scores"]):
                r["scores"] = [0.5662385]
            # Compute acc
            print(r["scores"])
            acc.append(r["scores"])

        print("acc: ", np.mean(acc))


if __name__ == "__main__":
    shoes = Shoes()
    shoes.prepare_data()
    shoes.load_pretrain_model()
    shoes.tarin()
    # shoes.detect()
    shoes.acc()