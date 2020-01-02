#-*-coding:utf-8-*-
import math
import os
import random
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours

from mrcnn.config import Config
#import utils
from mrcnn import model as modellib
from mrcnn import utils

from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

from visualize import random_colors, apply_mask


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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 8 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 832

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
        self.add_class("shapes", 1, "fig")
        self.add_class("shapes", 2, "grape")
        self.add_class("shapes", 3, "lemon")
        self.add_class("shapes", 4, "leaf")
        self.add_class("shapes", 5, "apple")
        self.add_class("shapes", 6, "orange")
        self.add_class("shapes", 7, "bitter")
        self.add_class("shapes", 8, "eggplant")



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
            if labels[i].find("fig") != -1:
                labels_form.append("fig")
            elif labels[i].find("grape") != -1:
                labels_form.append("grape")
            elif labels[i].find("lemon") != -1:
                labels_form.append("lemon")
            elif labels[i].find("leaf") != -1:
                labels_form.append("leaf")
            elif labels[i].find("apple") != -1:
                labels_form.append("apple")
            elif labels[i].find("orange") != -1:
                labels_form.append("orange")
            elif labels[i].find("bitter") != -1:
                labels_form.append("bitter")
            elif labels[i].find("eggplant") != -1:
                labels_form.append("eggplant")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


class Detect_Config(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



class Mask:
    def __init__(self, mask_order, class_id, class_name, mask, box, scores):
        self.mask_order = mask_order
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask
        self.box = box
        self.scores = scores
    #   self.min_rect
    #   self.min_box

    def get_min_rect(self):
        height = self.mask.shape[0]
        width = self.mask.shape[1]
        # print(height,width)

        points_of_mask = []
        for x in range(height):
            for y in range(width):
                if self.mask[x,y]:
                    points_of_mask.append([y, x])
        cnt = np.array(points_of_mask)
        rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        self.min_rect = rect
        min_box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        min_box = np.int0(min_box)
        self.min_box = min_box

class Point:
    def __init__(self,p):
        self.x = p[0]
        self.y = p[1]


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
        # self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "logs/shape20191128T1619/mask_rcnn_shape_0030.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        self.config = ShapesConfig()
        # self.config.display()

        self.dataset_root_path=self.ROOT_DIR+"/txl_data/"
        #self.dataset_root_path="/home/txl/PycharmProjects/Mask_RCNN_shoes/shoes_data/"
        self.img_floder = self.dataset_root_path + "pic"
        # print(img_floder)
        self.mask_floder = self.dataset_root_path + "cv2_mask"
        #yaml_floder = dataset_root_path
        self.imglist = os.listdir(self.img_floder)
        self.count = len(self.imglist)
        #print(self.imglist)
        #print(self.count)

        self.detectconfig = Detect_Config()

    def prepare_data(self):
        #train与val数据集准备
        self.dataset_train = DrugDataset()
        self.dataset_train.load_shapes(self.count, self.img_floder, self.mask_floder, self.imglist, self.dataset_root_path)
        self.dataset_train.prepare()
        self.dataset_val = DrugDataset()
        self.dataset_val.load_shapes(1, self.img_floder, self.mask_floder, self.imglist, self.dataset_root_path)
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
                    epochs=100,
                    layers='all')

    def detect(self):
        self.model = modellib.MaskRCNN(mode="inference", config=self.detectconfig,
                                  model_dir=self.MODEL_DIR)
        # SHARP_MODEL_PATH=os.path.join(SHARP_MODEL_DIR,"mask_rcnn_shapes_0000.h5")
        # self.SHARP_MODEL_PATH="/home/ljt/Mask_RCNN_shoes/logs/shape20190613T1601/mask_rcnn_shape_0010.h5"
        #self.SHARP_MODEL_PATH="../logs/shape20190613T1601/mask_rcnn_shape_0010.h5"
        self.SHARP_MODEL_PATH="../logs/shape20191130T1914/mask_rcnn_shape_0021.h5"
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
        image = cv2.imread("../txl_data/image/5.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.detect([image], verbose=1)
        r = results[0]


        self.get_all_mask_class_and_draw_minrect(r, image)

        # a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                                 self.dataset_val.class_names, r['scores'], show_bbox=False)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('contours.png', image)

    def get_all_mask_class_and_draw_minrect(self, r, image):
        num = r['masks'].shape[2]
        object = []

        _, ax = plt.subplots(1, figsize=(16, 16))
        shape = image.shape
        ax.set_ylim(shape[0] + 10, -10)
        ax.set_xlim(-10, shape[1] + 10)
        ax.axis('off')
        ax.set_title("hhhh")



        for i in range(num):  # create mask class for every instance
            obj = Mask(i, r['class_ids'][i], self.dataset_val.class_names[r['class_ids'][i]], r['masks'][:,:,i],
                       r['rois'][i], r['scores'][i])
            obj.get_min_rect()
            object.append(obj)
            # cv2.drawContours(image, [object[i].min_box], 0, (0, 0, 255), 3)
            print(object[i].class_name)

        for i in range(num):
            # if (object[i].class_name is "lemon") | (object[i].class_name is "grape")| (object[i].class_name is "fig"):
            shoe = object[i]
            center_x = shoe.min_rect[0][0]
            center_y = shoe.min_rect[0][1]
            width = shoe.min_rect[1][0]
            height = shoe.min_rect[1][1]
            theta = shoe.min_rect[2]
            PI = math.pi
            the = (theta / 180) * PI
            # print("object", i,":width height theta:", width, height, theta)

            if theta == 0:
                y = center_y - 3 * height / 4
                x = center_x
                point = (x, y)
                w_h = (width / 2, width / 4)
            elif width > height:
                x = center_x + (5 * width * math.cos(the)) / 8
                y = center_y + (5 * width * math.sin(the)) / 8
                # x = center_x - (2 * width * math.cos(the)) / 4
                # y = center_y - (2 * width * math.sin(the)) / 4
                point = (x, y)
                w_h = (height / 4, height / 2)
            else:
                x = center_x + (5 * height * math.sin(the)) / 8
                y = center_y - (5 * height * math.cos(the)) / 8
                # x = center_x - (2 * height * math.sin(the)) / 4
                # y = center_y + (2 * height * math.cos(the)) / 4
                point = (x, y)
                w_h = (width / 2, width / 4)
            # print("object", i, "center center_cut:", shoe.min_rect[0], center_x, center_y, x, y)

            rect = (point, w_h, theta)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (255, 0, 0), 2)



        colors = random_colors(num)
        masked_image = image.astype(np.uint32).copy()
        captions = ["lemon 0.988", "lemon 1.000", "lemon 0.995", "lemon 0.984", "lemon 0.992"]
        for i in range(num):
            # if (object[i].class_name is "lemon") | (object[i].class_name is "grape") | (object[i].class_name is "fig"):
            shoe = object[i]
            box = shoe.min_box
            width = shoe.min_rect[1][0]
            height = shoe.min_rect[1][1]
            theta = shoe.min_rect[2]
            score = shoe.scores
            label = shoe.class_name
            mask = shoe.mask

            p = patches.Rectangle(box[1], width, height, angle=theta, linewidth=2,  # add rectangle dash
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor="blue", facecolor='none')
            ax.add_patch(p)
            caption = "{} {:.3f}".format(label, score) if score else label  # add label
            # caption = captions[i]
            print(type(caption), caption)
            ax.text(box[0][0], box[0][1], caption,
                    color='w', size=8, backgroundcolor="blue")

            masked_image = apply_mask(masked_image, mask, colors[i])


        ax.imshow(masked_image.astype(np.uint8))
        plt.show()




        # with open('position.txt', 'w') as f:
        #     for i in range(num):
        #         if (object[i].class_name is "lemon") | (object[i].class_name is "grape"):
        #             for j in range(num):
        #                 if object[j].class_name == 'mouth':
        #                     if self.IsPointInMatrix(object[j].min_rect[0], object[i].min_box):
        #                         print("xiekou is belong shoes")
        #                         grasp_point, grasp_theta = self.draw_orientation(image, object[i], object[j])
        #
        #                         text = ""
        #                         for i in range(2):
        #                             text += str(grasp_point[i]) + " "
        #                         text += str(grasp_theta) + "\n"
        #                         f.writelines(text)


    # def draw_orientation(self, image, shoe, belt):
    #     p1 = tuple([int(x) for x in belt.min_rect[0]])  # xiekou ceter
    #     p2 = tuple([int(x) for x in shoe.min_rect[0]])  # shoe center
    #     width = shoe.min_rect[1][0]
    #     height = shoe.min_rect[1][1]
    #     theta = shoe.min_rect[2]
    #
    #     if width < height:
    #         x = p1[0] + width * math.cos(theta) / 2
    #         y = p1[1] + width * math.sin(theta) / 2
    #         grasp_point = (x, y)
    #         p3 = self.middle_point(shoe.min_box[1], shoe.min_box[2])
    #         p4 = self.middle_point(shoe.min_box[0], shoe.min_box[3])
    #         if self.distance(p1, p3) > height / 2:
    #             cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
    #             grasp_theta = theta
    #         else:
    #             cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)
    #             grasp_theta = 180 + theta
    #     else:
    #         x = p1[0] + height * math.sin(theta) / 2
    #         y = p1[1] - height * math.cos(theta) / 2
    #         grasp_point = (x, y)
    #         p3 = self.middle_point(shoe.min_box[2], shoe.min_box[3])
    #         p4 = self.middle_point(shoe.min_box[0], shoe.min_box[1])
    #
    #         if self.distance(p1, p3) > width / 2:
    #             cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
    #             grasp_theta = 90 + theta
    #         else:
    #             cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)
    #             grasp_theta = -90 + theta
    #     cv2.imwrite('contours.png', image)
    #
    #     return grasp_point,grasp_theta



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
    # shoes.load_pretrain_model()
    # shoes.tarin()
    shoes.detect()
    # shoes.acc()