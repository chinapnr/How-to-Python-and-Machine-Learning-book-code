# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:32:20 2017

@author: yi.xiong
"""

# 1. 切割出有效期片段
# 2. 以宽高比判断是正规日期格式还是带长期格式
# 3. 图像灰度 - 增强 - 二值化 （去噪预处理）
# 4. 按照两种格式的规则分别找出日期中间的横线，将图像分割为三部分，日期 - 日期
# 5. 先切割出年和日期，然后分别切割出单个字符

import sys
sys.path.append('../')

import os
import shutil
import cv2
import csv

import valid_recognition.Preprocess_Image as PI
import valid_recognition.Split_Image as SI
import valid_recognition.Cut_Year_Date as YD
import valid_recognition.prepare_cut_character_v2 as prepare

def resize_img(img):
    row,columns = img.shape[:2]
    ratio = 40 / row
    img = cv2.resize(img,(int(columns*ratio),int(row*ratio)))
    return img

# 根据txt文件中的有效期坐标得到左右边界
def read_coordinate(file):
    with open(file, "r", encoding= 'utf8') as f:
        mark_start = False
        mark_end = False
        previous_line = ''
        lines = f.readlines()
        for line in lines:
            if mark_end == True:
                points = previous_line.split(',')
                end= [int(points[0])+int(points[2]), int(points[1])+int(points[3])]
                return start, end
            if mark_start == True:
                start = [int(line.split(',')[0]), int(line.split(',')[1])]
                mark_start = False
            if line == "[expire]\n":
                mark_start = True
            if line in ['[police_characters]\n','[expire_characters]\n']:
                mark_end = True
            if mark_end == False:
                previous_line = line            
    return start, end

# 切割出有效期区域
def cut_area(file, start, end):
    img = cv2.imread(file)
    return img[start[1]:end[1],start[0]:end[0],:]


# 根据宽高比判断是正规长度格式还是带长期的格式
def calculate_range(img):
    height = img.shape[0]
    width = img.shape[1]
    ratio = round(width * 1.0 / height,1)   # 宽高比保留至1位小数
    if ratio>= 7:   # sdk里此值为9.8，针对带背景身份证，此值改为7
        return 'regular'  # xxxx.xx.xx - xxxx.xx.xx
    else:
        return 'abnormal'  # xxxx.xx.xx - 长期
  
# 图像灰度 - 增强 - 二值化 （去噪预处理）
def preprocess_img(scr):
    # 灰度化
    img_Gray = PI.gray_img(scr)
    # 图像增强 - 直方图均衡化
    img_Enhance = PI.enhance_img(img_Gray)
    # 二值化
    img_Binary = PI.binary_img(img_Enhance)
    return img_Binary

def do_wholeprocess_valid(line_src_img):
    img_area = resize_img(prepare.prepare_character(line_src_img))
    # 2. 根据图像宽高比判断有效期格式类型
    label = calculate_range(img_area)
    # print (label)
    # 3. 将得到的有效期图像进行灰度 - 增强 - 二值化 （去噪预处理）
    # 用于确定切割点
    img_area_preprocess = preprocess_img(img_area)
    # 4. 分别对正规格式和带长期的图像进行分割，得到xxxx.xx.xx
    if label == 'regular':
        imgs_cut = SI.cut_regular(img_area_preprocess, img_area)  # 2
    else:
        imgs_cut = SI.cut_abnormal(img_area_preprocess, img_area)  # 1
    # 5.将xxxx.xx.xx预处理，切割出单个字符
    if imgs_cut != 'error':
        imgs_cut_preprocess = []
        for img_cut in imgs_cut:
            imgs_cut_preprocess.append(preprocess_img(img_cut))
        single_imgs = YD.cut_year_date(imgs_cut_preprocess, imgs_cut)
        if single_imgs != 'error':
            for single_img in single_imgs:
                x, y, z = single_img.shape
                if x == 0 or y == 0 or z == 0:
                    wrong_list.append(folder)
                    single_imgs = 'error'
                    break
            if single_imgs != 'error':
                return single_imgs
        else:
            return None
    else:
        return None






















