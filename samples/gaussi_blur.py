# # -*- coding:utf-8 -*-
# """数据增强
#    1. 翻转变换 flip
#    2. 随机修剪 random crop
#    3. 色彩抖动 color jittering
#    4. 平移变换 shift
#    5. 尺度变换 scale
#    6. 对比度变换 contrast
#    7. 噪声扰动 noise
#    8. 旋转变换/反射变换 Rotation/reflection
#    author: XiJun.Gong
#    date:2016-11-29
# """
#
# from PIL import Image, ImageEnhance, ImageOps, ImageFile
# import numpy as np
# import random
# import threading, os, time
# import logging
# import numpy as np
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#
# import cv2
#
# logger = logging.getLogger(__name__)
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#
# class DataAugmentation:
#     """
#     包含数据增强的八种方式
#     """
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def openImage(image):
#         return Image.open(image, mode="r")
#
#     @staticmethod
#     def randomRotation(image, mode=Image.BICUBIC):
#         """
#          对图像进行随机任意角度(0~360度)旋转
#         :param mode 邻近插值,双线性插值,双三次B样条插值(default)
#         :param image PIL的图像image
#         :return: 旋转转之后的图像
#         """
#         random_angle = np.random.randint(1, 360)
#         return image.rotate(random_angle, mode)
#
#     @staticmethod
#     def randomCrop(image):
#         """
#         对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
#         :param image: PIL的图像image
#         :return: 剪切之后的图像
#         """
#         image_width = image.size[0]
#         image_height = image.size[1]
#         crop_win_size = np.random.randint(40, 68)
#         random_region = (
#             (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
#             (image_height + crop_win_size) >> 1)
#         return image.crop(random_region)
#
#     @staticmethod
#     def randomColor(image):
#         """
#         对图像进行颜色抖动
#         :param image: PIL的图像image
#         :return: 有颜色色差的图像image
#         """
#         random_factor = np.random.randint(0, 31) / 10.  # 随机因子
#         color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
#         random_factor = np.random.randint(10, 21) / 10.  # 随机因子
#         brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
#         random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
#         contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
#         random_factor = np.random.randint(0, 31) / 10.  # 随机因子
#         return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
#
#     @staticmethod
#     def randomGaussian(image, mean=0.2, sigma=0.3):
#         """
#          对图像进行高斯噪声处理
#         :param image:
#         :return:
#         """
#
#         def gaussianNoisy(im, mean=0.2, sigma=0.3):
#             """
#             对图像做高斯噪音处理
#             :param im: 单通道图像
#             :param mean: 偏移量
#             :param sigma: 标准差
#             :return:
#             """
#             for _i in range(len(im)):
#                 im[_i] += random.gauss(mean, sigma)
#             return im
#
#         # 将图像转化成数组
#         img = np.asarray(image)
#         img = np.require(img, dtype='f4', requirements=['O', 'W'])
#         img.flags.writeable = True  # 将数组改为读写模式
#         width, height = img.shape[:2]
#         img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
#         img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
#         img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
#         img[:, :, 0] = img_r.reshape([width, height])
#         img[:, :, 1] = img_g.reshape([width, height])
#         img[:, :, 2] = img_b.reshape([width, height])
#         return Image.fromarray(np.uint8(img))
#
#     @staticmethod
#     def saveImage(image, path):
#         image.save(path)
#
#
# def makeDir(path):
#     try:
#         if not os.path.exists(path):
#             if not os.path.isfile(path):
#                 # os.mkdir(path)
#                 os.makedirs(path)
#             return 0
#         else:
#             return 1
#     except Exception as e:
#         print(str(e))
#         return -2
#
#
# def imageOps(func_name, image, des_path, file_name, times=5):
#     funcMap = {"randomRotation": DataAugmentation.randomRotation,
#                # "randomCrop": DataAugmentation.randomCrop,
#                "randomColor": DataAugmentation.randomColor,
#                # "randomGaussian": DataAugmentation.randomGaussian
#                }
#     if funcMap.get(func_name) is None:
#         logger.error("%s is not exist", func_name)
#         return -1
#
#     for _i in range(0, times, 1):
#         new_image = funcMap[func_name](image)
#         DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))
#
#
# opsList = {"randomRotation",  "randomColor"} # "randomCrop",, "randomGaussian"
#
#
# def threadOPS(path, new_path):
#     """
#     多线程处理事务
#     :param src_path: 资源文件
#     :param des_path: 目的地文件
#     :return:
#     """
#     if os.path.isdir(path):
#         img_names = os.listdir(path)
#     else:
#         img_names = [path]
#     for img_name in img_names:
#         print
#         img_name
#         tmp_img_name = os.path.join(path, img_name)
#         if os.path.isdir(tmp_img_name):
#             if makeDir(os.path.join(new_path, img_name)) != -1:
#                 threadOPS(tmp_img_name, os.path.join(new_path, img_name))
#             else:
#                 print
#                 'create new dir failure'
#                 return -1
#                 # os.removedirs(tmp_img_name)
#         elif tmp_img_name.split('.')[1] != "DS_Store":
#             # 读取文件并进行操作
#             image = DataAugmentation.openImage(tmp_img_name)
#             threadImage = [0] * 5
#             _index = 0
#             for ops_name in opsList:
#                 threadImage[_index] = threading.Thread(target=imageOps,
#                                                        args=(ops_name, image, new_path, img_name,))
#                 threadImage[_index].start()
#                 _index += 1
#                 time.sleep(0.2)
#
#
# if __name__ == '__main__':
#     threadOPS("/home/txl/train/12306train",
#               "/home/txl/train/12306train3")
#
#     # img = cv2.imread('/home/txl/train/12306train/13.jpg')
#     # img0 = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)
#     # img1 = cv2.flip(img, 0)  # 垂直翻转
#     # img2 = cv2.flip(img, 1)  # 水平翻转
#     #
#     #
#     # # cv2.imshow('Source image', img)
#     # # cv2.imshow('blur image', img0)
#     # # cv2.imshow('chuizhi', img1)
#     # # cv2.imshow('shuiping', img2)
#     # cv2.imwrite('/home/txl/train/txl/img.jpg', img)
#     # cv2.imwrite('/home/txl/train/txl/img0.jpg', img0)
#     # cv2.imwrite('/home/txl/train/txl/img1.jpg', img1)
#     # cv2.imwrite('/home/txl/train/txl/img2.jpg', img2)
#     #
#     #
#     # cv2.waitKey()
import os
import os.path
from PIL import Image, ImageEnhance
from skimage import io
import random
import numpy as np
import shutil

def salt_and_pepper_noise(img, proportion=0.05):
    noise_img =img
    height,width =noise_img.shape[0],noise_img.shape[1]
    num = int(height*width*proportion)#多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w] =0
        else:
            noise_img[h,w] = 255
    return noise_img

def gauss_noise(image):
    img = image.astype(np.int16)  # 此步是为了避免像素点小于0，大于255的情况
    mu = 0
    sigma = 25
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

if __name__ == '__main__':
    path = "/home/txl/train/image/"
    path1 = "/home/txl/train/gauss_image/"
    path2 = "/home/txl/train/flip_vertical_image/"
    path3 = "/home/txl/train/flip_level_image/"
    path4 = "/home/txl/train/rotate_image/"
    path5 = "/home/txl/train/color_image/"
    path6 = "/home/txl/train/all_image/"

    # for item in os.listdir(path):
    #     img = io.imread(path + item)
    #     # noise_img = salt_and_pepper_noise(img)
    #     gauss_img = gauss_noise(img)
    #     # io.imshow(gauss_img)
    #     # io.show()
    #     io.imsave(path1 + item, gauss_img)


    # for item in os.listdir(path):
    #     img = Image.open(path + item)
    #     im_rotate = img.rotate(135)
    #     # im_rotate.show()
    #     im_rotate.save(path4+item)
    # for item in os.listdir(path):
    #     img = Image.open(path+item)
    #     im_flip1 = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     im_flip2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     im_flip1.save(path3+item)
    #     im_flip2.save(path2+item)
    # for item in os.listdir(path):
    #     img = Image.open(path + item)
    #     im_color = randomColor(img)
    #     # im_color.show()
    #     im_color.save(path5 + item)

    # paths = [path1, path2, path3, path4]
    paths = [path5]
    count = 409
    for item in paths:
        for i in os.listdir(item):
            count+=1
            shutil.move(item + i, os.path.join(path6, str(count)+".jpg"))



