# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import hog

def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

def prepare_character(src_img, threshold_value=0.05, peak_width_threshold=30):
    '''
    此脚本用于处理cut_line_scrpt_v4产生的切行样本， 去除行周围（上下左右）的空白部分，使得样本可以作为cut_character_v1的输入进行切字。
    
    主要思想:
    使用HOG特征（检测文字的纹理），先左右切割，切出文字区域（HOG特征并且水平膨胀10像素，使得进行垂直投影时文字区域是一个大峰）找出峰宽大于30的区域作为文字区域
    合并获得完整的文字区域，随后使用HOG特征水平投影切除上下区域的空白
    
    :param src_img: 原始图像 (cut_line_script_v4 output)
    :param threshold_value: 上下切割时使用的阈值
    :param peak_width_threshold: 左右切割时使用的峰宽阈值. (用于去噪)
    :return: 切除空白后的行样本（cut_character_v1 input)
    '''
    # 备份原始图片
    img = src_img.copy()
    # 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 标准化
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # hog特征 , pixels_per_cell与特征尺寸相关
    fd, hog_img = hog(img,orientations=4,pixels_per_cell=(3,3), visualise=True)
    # 转化为UINT8类型
    hog_img = np.asarray(hog_img, dtype='uint8')
    # 标准化特征
    cv2.normalize(hog_img, hog_img, 0, 255, cv2.NORM_MINMAX)
    # 二值化
    res, binary_img = cv2.threshold(hog_img, 100, 255, cv2.THRESH_BINARY)

    # 水平膨胀10像素
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    binary_img = cv2.dilate(binary_img, element, iterations = 1)

    # 开始分割
    # 垂直投影
    x_map = np.mean(binary_img, axis=0)
    x_map /= np.max(x_map)
    
    ''' 投影具有几个峰 (每个峰有可能是噪声或者是字，需要通过峰宽来判断) '''
    # 找出字符的边界
    # 左右加三个黑色像素
    tmp_x_map = np.append(np.append([0, 0, 0], x_map), [0, 0, 0])
    # 找出所有峰与峰之间的间隔位置
    x_map_candidate = np.where(tmp_x_map == 0)[0]

    # 找出每个峰的左右两侧的点
    peak_left_pts = []
    peak_right_pts = []
    temp = []
    for pt in x_map_candidate:
        if len(temp) == 0:
            temp.append(pt)
        elif pt - temp[-1] <= 3:
            temp.append(pt)
        elif pt - temp[-1] > 3:
            peak_left_pts.append(temp[-1] - 3)
            peak_right_pts.append(temp[0] - 3)
            temp = [pt]
    # 增加最后剩余的两个点
    peak_left_pts.append(temp[-1] - 3)
    peak_right_pts.append(temp[0] - 3)
    # 去除开始和结尾部分的点
    peak_left_pts.remove(peak_left_pts[-1])
    peak_right_pts.remove(peak_right_pts[0])

    # 过滤噪声峰
    region_width_list = []
    updated_peak_left = []
    updated_peak_right = []
    # 当但钱的峰宽大于设定的峰宽阈值时，记录当前峰宽、峰起始点和终止点
    for i in range(len(peak_left_pts)):
        current_width = peak_right_pts[i] - peak_left_pts[i]
        if current_width > peak_width_threshold:
            region_width_list.append(current_width)
            updated_peak_left.append(peak_left_pts[i])
            updated_peak_right.append(peak_right_pts[i])
    # 如果没有峰（即空白行），返回空
    if len(region_width_list) == 0:
        return None
    # 切除文字区域并减小2个像素（由于之前横向膨胀了10个像素，这边减两个像素，防止左右空白太大）
    left_point = updated_peak_left[0] + 2
    right_point = updated_peak_right[-1] -2
    left_right_cutted_img = src_img[:, left_point:right_point]
    # 灰度化
    img = cv2.cvtColor(left_right_cutted_img, cv2.COLOR_BGR2GRAY)
    # 标准化
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    try:
        # 再次HOG特征
        fd, hog_img = hog(img, orientations=4, pixels_per_cell=(3, 3), visualise=True)
        # 转化为UINT8类型
        hog_img = np.asarray(hog_img, dtype='uint8')
        # 标准化特征
        cv2.normalize(hog_img, hog_img, 0, 255, cv2.NORM_MINMAX)
        # 二值化
        res, binary_img = cv2.threshold(hog_img, 100, 255, cv2.THRESH_BINARY)
        # 水平投影
        y_map = np.mean(binary_img, axis=1)
        y_map /= np.max(y_map)

        # 防错
        try:
            # 下边界为峰的第一个点
            down_point = np.min(np.where(y_map > threshold_value)[0])
        except:
            # 如果出错则让下边界为0
            down_point = 0
        try:
            # 上边界为峰的最后一个点
            up_point = np.max(np.where(y_map > threshold_value)[0])
        except:
            # 如果出错则让上边界为高度
            up_point = len(y_map) -1

        # 调整多一个像素（防止切掉太多）
        down_point = down_point - 1 if down_point > 0 else down_point
        up_point = up_point + 1 if up_point < src_img.shape[0] - 1 else up_point
        # 返回结果
        return src_img[down_point:up_point, left_point:right_point]
    except:
        # 防错如果产生错误则返回空
        return None
