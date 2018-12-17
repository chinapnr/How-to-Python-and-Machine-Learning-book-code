# -*- coding:utf-8 -*-


import numpy as np
import os
import cv2
import shutil
#from matplotlib import pyplot as plt


def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def calc_rate(line):
    x1, y1, x2, y2, x3, y3, x4, y4 = line.split(',')
    width = abs(float(x2) - float(x1))
    height = abs(float(y1) - float(y4))
    return height, [int(d) for d in (x1, y1, x3, y3)]
    # return width*height, height, height/width

# def do_cut_back_v2(src_path, f):
def do_cut_back(img, line_box):
    line_height_list = []
    line_pts_list = []

    return_result = []

    # with open(os.path.join(src_path, f), 'r') as reader:
    # lines = reader.readlines()

    lines = [','.join([str(e) for e in d]) for d in line_box]
    if len(lines) != 3:
        # print(f)
        return None

    for line in lines:
        h, pts = calc_rate(line)
        line_height_list.append(h)
        line_pts_list.append(pts)
    height_max_index = np.argmax(line_height_list)
    x1, y1, x3, y3 = line_pts_list[height_max_index]
    all_y1_list = [d[1] for d in line_pts_list]

    little_y1_index_list = [i for i, d in enumerate(all_y1_list) if d > y1]
    # img = cv2.imread(os.path.join(src_path, f.replace('.txt', '.png')), 1)

    if len(little_y1_index_list) == 2:

        current_y1_list = [all_y1_list[d] for d in little_y1_index_list]
        current_y1_list_index = np.argsort(current_y1_list)

        for save_index, i in enumerate([little_y1_index_list[d] for d in current_y1_list_index]):
            current_x1, current_y1, current_x3, current_y3 = line_pts_list[i]
            return_result.append(img[current_y1 - 2: current_y3, (current_x1 - 2): (current_x3 + 8)])
    elif len(little_y1_index_list) == 0:
        useful_line_index = [d for d in range(3) if d not in [height_max_index]]

        current_y1_list = [all_y1_list[d] for d in useful_line_index]
        current_y1_list_index = np.argsort(current_y1_list).tolist()[::-1]

        for save_index, i in enumerate([useful_line_index[d] for d in current_y1_list_index]):
            current_x1, current_y1, current_x3, current_y3 = line_pts_list[i]
            line_img = img[current_y1: current_y3, (current_x1 - 2): (current_x3 + 8)]
            line_img = rotate(line_img, 180)
            return_result.append(line_img)
    return return_result

