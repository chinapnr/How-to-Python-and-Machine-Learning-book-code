# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:12:30 2017

@author: li.kou
"""

import sys
sys.path.append('../')

import numpy as np
import cv2
import os
from math import fabs, sin, cos, radians

# MTCNN
import tensorflow as tf
# import detect_face
from rotate_cert import detect_face
import time
import numpy as np

# global init

# sess = tf.Session()
# pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 40  # minimum size of face
threshold = [0.6, 0.7, 0.9]  # three steps's threshold
factor = 0.709  # scale factor

# end global init

def check_dir(path):
    if not(os.path.isdir(path) and os.path.exists(path)):
        os.mkdir(path)

def detectFace(imgscr, models):
    # global sess, pnet, rnet, onet, minsize, threshold, factor
    global minsize, threshold, factor
    pnet, rnet, onet = models
    # modified
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

    for angle in [0, 90, 180, 270]:
        img_rotated = rotate_img_alter(imgscr, angle)
        img_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        bounding_boxes, points = detect_face.detect_face(img_rotated, minsize, pnet, rnet, onet, threshold, factor)

        ### very important reset tensorflow session ###
        tf.reset_default_graph()

        if len(bounding_boxes) > 0:
            return len(bounding_boxes), cv2.cvtColor(img_rotated, cv2.COLOR_RGB2BGR)

    for angle in [0,90,180,270]:
        img_rotated = rotate_img_alter(imgscr,angle)
        gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5) #1.2和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        if len(faces) > 0:
            return len(faces), img_rotated
    return 0, imgscr

# # 将原图调整为正方形
# def adjustsize_img(img):
#     width = img.shape[1]  #图像宽度
#     height = img.shape[0]  #图像高度
#     channel = img.shape[2]
#     if height > width:
#         size = height
#     else:
#         size = width
#     # 创建一个空白的新图片，尺寸为size*size*size
#     newimg = np.zeros((size,size,channel), np.uint8)
#     # 放入size*size大小图片的起始位置
#     offset_height = int(np.ceil((size - height) / 2))
#     offset_width = int(np.ceil((size - width) / 2))
#     # 将调整比例后的图片内容复制到空白图片
#     for x in range(height):
#         for y in range(width):
#             for i in range(channel):
#                newimg[x +offset_height, y +offset_width,i] = img[x, y,i]
#     # 返回预处理完成后的图片
#     return newimg
#
#
# def rotate_img(imgscr,angle):
#     if angle == 0:
#         return imgscr
#     height,width = imgscr.shape[:2]
#     if height != width:
#         img_adjusted = adjustsize_img(imgscr)
#     else:
#         img_adjusted = imgscr
#     height,width = img_adjusted.shape[:2]
#
#     #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
#     M = cv2.getRotationMatrix2D((width/2,height/2),angle,1)
#     return cv2.warpAffine(img_adjusted,M,(width,height)) #第三个参数：变换后的图像大小 width * height

def rotate_img_alter(img, angle):#try this...
    height,width=img.shape[:2]

    heightNew=int(width*fabs(sin(radians(angle)))+height*fabs(cos(radians(angle))))
    widthNew=int(height*fabs(sin(radians(angle)))+width*fabs(cos(radians(angle))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),angle,1)

    matRotation[0,2] +=(widthNew-width)/2
    matRotation[1,2] +=(heightNew-height)/2

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    return imgRotation

if __name__ == '__main__':
    src_image_path = r'C:\Users\li.kou\Desktop\test_front'
    dst_image_path = r'C:\Users\li.kou\Desktop\dst'
    error_image_path = r'C:\Users\li.kou\Desktop\error'

    check_dir(dst_image_path)
    check_dir(error_image_path)

    consume_time_list = []

    for f in os.listdir(src_image_path):
        start_time = time.time()
        face, img = detectFace(cv2.imread(os.path.join(src_image_path, f), 1))
        consume_time_list.append(time.time() - start_time)
        # if len(face) < 1:
        if face < 1:
            print('error : {}'.format(f))
            cv2.imwrite(os.path.join(error_image_path, f), img)
        else:
            print('finish : {}'.format(f))
            cv2.imwrite(os.path.join(dst_image_path, f), img)

    print('average consume time: {}'.format(np.mean(consume_time_list)))


        
        
        
        
        
        
        
        
        
        
        
        