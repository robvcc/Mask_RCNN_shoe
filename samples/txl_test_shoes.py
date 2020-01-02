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
import math

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
    NUM_CLASSES = 1 + 3  # background + 3 shapes

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
        self.add_class("shapes", 1, "shoes")
        self.add_class("shapes", 2, "shoes1")
        self.add_class("shapes", 3, "belt")


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
            if labels[i].find("shoes") != -1:
                labels_form.append("shoes")
            elif labels[i].find("shoes1") != -1:
                labels_form.append("shoes1")
            elif labels[i].find("belt") != -1:
                labels_form.append("belt")




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
        # self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "logs/shape20191113T1047/mask_rcnn_shape_0022.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        self.config = ShapesConfig()
        # self.config.display()

        self.dataset_root_path=self.ROOT_DIR+"/txl_shoes_data/"
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
        self.dataset_val.load_shapes(10, self.img_floder, self.mask_floder, self.imglist, self.dataset_root_path)
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
        self.SHARP_MODEL_PATH="../logs/shape20191113T1047/mask_rcnn_shape_0029.h5"
        self.model.load_weights(self.SHARP_MODEL_PATH, by_name=True)
        print(self.SHARP_MODEL_PATH)

        # import skimage
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
        image = cv2.imread("../shoes_data/test/shoes1.jpg")
        results = self.model.detect([image], verbose=1)

        r = results[0]
        mask = r['masks']

        # print(mask.shape[0],mask.shape[1],mask.shape[2])
        # print(r['rois'], type(r['rois']))
        # print(r['class_ids'], type(r['class_ids'][0]))
        print(self.dataset_val.class_names, type(self.dataset_val.class_names[0]))
        # print(r['scores'], type(r['scores'][0]))

        # self.mask_to_min_rect(mask, image)
        self.get_all_mask_class_and_draw_minrect(r, image)
        a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                         self.dataset_val.class_names, r['scores'], show_bbox=False)


    def get_all_mask_class_and_draw_minrect(self, r, image):
        num = r['masks'].shape[2]
        object = []
        for i in range(num):  # create mask class for every instance
            obj = Mask(i, r['class_ids'][i], self.dataset_val.class_names[r['class_ids'][i]], r['masks'][:,:,i],
                       r['rois'][i], r['scores'][i])
            obj.get_min_rect()
            object.append(obj)
            cv2.drawContours(image, [object[i].min_box], 0, (0, 0, 255), 3)
            print(object[i].class_name)

        for i in range(num):
            if (object[i].class_name is "shoes") | (object[i].class_name is "shoes1"):
                for j in range(num):
                    if object[j].class_name == 'belt':
                        if self.IsPointInMatrix(object[j].min_rect[0],object[i].min_box):
                            print("xiekou is belong shoes")
                            self.draw_orientation(image, object[i], object[j])



    def IsPointInMatrix(self, center_p, min_box):
        p = Point(center_p)
        p1 = Point(min_box[0])
        p2 = Point(min_box[1])
        p3 = Point(min_box[2])
        p4 = Point(min_box[3])
        return self.GetCross(p1, p2, p) * self.GetCross(p3, p4, p) >= 0 and self.GetCross(p2, p3, p) * self.GetCross(p4, p1, p) >= 0

    def GetCross(self, p1, p2, p):
        return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)

    def mask_to_min_rect(self, mask, image):
        tolist = mask.tolist()

        height = len(tolist)
        width = len(tolist[0])
        object_num = len(tolist[0][0])
        # height = mask.shape[0]
        # width = mask.shape[1]
        # object_num = mask.shape[2]

        print(height, width, object_num)

        points = []

        for n in range(object_num):
            point_of_n = []
            for x in range(height):
                for y in range(width):
                    if tolist[x][y][n]:
                        point_of_n.append([y, x])
            points.append(point_of_n)
        # print(points[0])
        num = len(points)  #
        print(num)
        boxs = []
        with open('position.txt', 'w') as f:
            for i in range(len(points)):
                cnt = np.array(points[i])
                rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                print(rect)
                box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
                box = np.int0(box)
                boxs.append(box)
                print(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
                # cv2.imwrite('contours.png', image)

                text = ""
                for i in range(2):
                    for item in rect[i]:
                        text = text + str(item) + " "
                text = text + str(rect[2]) + "\n"
                f.writelines(text)
        self.draw_shoes_orientation(image,boxs)
        # with open('position.txt', 'r') as f:
        #     poi = []
        #     lines = f.readlines()
        #     for line in lines:
        #         min_rect = line.split()
        #         for i in range(len(min_rect)):
        #             min_rect[i] = float(min_rect[i])
        #         poi.append(int(min_rect[0]))
        #         poi.append(int(min_rect[1]))
        #     cv2.arrowedLine(image, (poi[2], poi[3]), (poi[0], poi[1]), (255, 0, 0), 5)
        #     cv2.imwrite('contours.png', image)

    def draw_orientation(self, image, shoe, belt):
        p1 = tuple([int(x) for x in belt.min_rect[0]])  # xiekou ceter
        p2 = tuple([int(x) for x in shoe.min_rect[0]])  # shoe center
        width = shoe.min_rect[1][0]
        height = shoe.min_rect[1][1]

        if width < height:
            p3 = self.middle_point(shoe.min_box[1], shoe.min_box[2])
            p4 = self.middle_point(shoe.min_box[0], shoe.min_box[3])
            if self.distance(p1, p3) > height / 2:
                cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
            else:
                cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)
        else:
            p3 = self.middle_point(shoe.min_box[2], shoe.min_box[3])
            p4 = self.middle_point(shoe.min_box[0], shoe.min_box[1])

            if self.distance(p1, p3) > width / 2:
                cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
            else:
                cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)
        cv2.imwrite('contours.png', image)

    def draw_shoes_orientation(self, image, boxs):
        with open('position.txt', 'r') as f:
            min_rects=[]
            poi=[]
            lines = f.readlines()
            for line in lines:
                min_rect = line.split()
                for i in range(len(min_rect)):
                    min_rect[i] = float(min_rect[i])
                poi.append(int(min_rect[0]))
                poi.append(int(min_rect[1]))
                min_rects.append(min_rect)
            width = min_rects[0][2]
            height = min_rects[0][3]
            if width < height:
                p1 = (poi[2], poi[3])  # xiekouzhongxin
                p2 = (poi[0], poi[1])  # xiezizhongxin
                p3 = self.middle_point(boxs[0][1], boxs[0][2])
                p4 = self.middle_point(boxs[0][0], boxs[0][3])

                if self.distance(p1,p3) > height/2:
                    cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
                else:
                    cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)
            else:
                p1 = (poi[2], poi[3])  # xiekouzhongxin
                p2 = (poi[0], poi[1])  # xiezizhongxin
                p3 = self.middle_point(boxs[0][2], boxs[0][3])
                p4 = self.middle_point(boxs[0][0], boxs[0][1])

                if self.distance(p1, p3) > width / 2:
                    cv2.arrowedLine(image, p2, p3, (255, 0, 0), 5)
                else:
                    cv2.arrowedLine(image, p2, p4, (255, 0, 0), 5)


            cv2.imwrite('contours.png', image)

    def middle_point(self,point1,point2):
        x = (point1[0]+point2[0])/2
        y = (point1[1]+point2[1])/2
        return (int(x), int(y))

    def distance(self,point1,point2):
        dis = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
        return dis

if __name__ == "__main__":
    shoes = Shoes()
    shoes.prepare_data()
    shoes.detect()