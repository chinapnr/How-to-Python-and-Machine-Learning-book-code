# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:58:06 2017

@author: yi.xiong
"""


import cv2  
import os
import numpy as np
import math
import shutil
import time

import sys
sys.path.append('../')

import number_recognition.Preprocess as pr
import number_recognition.TextLine_Index as ti


def segment_number(imgscr):
    row, column = imgscr.shape[:2]

    resize_ratio = 35. / row

    imgscr = cv2.resize(imgscr, (int(column * resize_ratio), 35))

    textline, rects = ti.detect_textline(pr.preprocess_full_img(imgscr), imgscr)

    try:
        if len(textline) > 18:
            print("more than 18")
        elif len(textline) < 18:
            print("less than 18")
        if len(textline) != 18:
            print("识别卡号不准")
        # 保存身份证号
        w_list = [text[2] for text in textline[len(textline) - 18:]]
        mean_w = np.mean(w_list)
        return_img_list = []
        for i, text in enumerate(textline[len(textline) - 18:]):
            x, y, w, h = text
            if w < int(mean_w):
                diff = int((int(mean_w) - w) / 2)
                x = x - diff
                w = 2 * diff + w
            return_img_list.append(imgscr[y:y + h, x:x + w, :])
    except:
        print('error happened in {}'.format(file))
        return None
    return return_img_list

