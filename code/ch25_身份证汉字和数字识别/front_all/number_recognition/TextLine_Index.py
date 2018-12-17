# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:12:30 2017

@author: yi.xiong
"""

import cv2  
import numpy as np
import math

# debugger
import time

# 去掉太大和太小的框
def rect(x, y, w, h, img_preprocessed, contours):
    #### current remove those two condition #####
    # if w > img_preprocessed.shape[1] * 0.1 or h > img_preprocessed.shape[0] * 0.2:
    #     print('condition 1')
    #     return False
    # if w < 10 or h < 10:
    #     print('condition 2')
    #     return False
    #############################################
    aspect_ratio = min(w / h, h / w)
    if aspect_ratio < 0.3 or aspect_ratio > 1:
        return False
    pixels = 0
    box = img_preprocessed[y:y + h, x:x + w]
    for line in box:
        for pixel in line:
            if pixel == 0:
                pixels = pixels + 1
    occupation_ratio = pixels / (w * h)
    if occupation_ratio < 0.1 or occupation_ratio > 0.8:
        return False
    return True


# 去掉嵌套的框
def filter_rect(rects):
    result = []
    for rect1 in rects:
        contain = False
        x1, y1, w1, h1 = rect1
        for rect2 in rects:
            if rect2 != rect1:
                x2, y2, w2, h2 = rect2
                if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                    contain = True
                    break
        if not contain:
            result.append(rect1)
    return result


# 将属于同一文本相邻的字符合成一对
def component_link(rects, parameter):
    pairs = []
    for i, rect in enumerate(rects[:len(rects) - 1]):
        x, y, w_start, h_start = rect
        x_start = x + w_start / 2
        y_start = y + h_start / 2
        for point in rects[i + 1:]:
            x, y, w, h = point
            x_c = x + w / 2
            y_c = y + h / 2
            dist = np.sqrt((x_start - x_c) ** 2 + (y_start - y_c) ** 2)
            if (x_start - x_c) ** 2 >= (y_start - y_c) ** 2 * 4:
                if w / w_start >= 0.05 and w / w_start <= 20 and h / h_start >= 0.05 and h / h_start <= 20:
                    # if  dist < (w+w_start)*parameter:
                    if dist < (h + h_start) * 0.88 and dist < (w + w_start) * parameter:
                        pairs.append([rect, point])

    return pairs


def unique(pair, chains):
    for chain in chains:
        for point in chain:
            if point not in pair:
                pair.append(point)
    return pair


# whether two chains have at least one commone point
def have_common(pair, chain):
    p1, p2 = pair
    if p1 in chain or p2 in chain:
        return True
    else:
        return False


def calculate_line(p1, p2):
    x1, y1, w1, h1 = p1
    x2, y2, w2, h2 = p2
    # 竖直直线  x-c = 0
    if x1 + w1 / 2 == x2 + w2 / 2:
        return 1, 0
    # 水平直线  y-c = 0
    elif y1 + h1 / 2 == y2 + h2 / 2:
        return 0, 1
    # kx - y + c = 0
    else:
        return (y2 + h2 / 2 - y1 - h1 / 2) / (x2 + w2 / 2 - x1 - w1 / 2), -1


def get_angle(chain1, chain2):
    p1 = chain1[0]
    p2 = chain1[-1]
    p3 = chain2[0]
    p4 = chain2[-1]
    a1, b1 = calculate_line(p1, p2)
    a2, b2 = calculate_line(p3, p4)
    angle = math.acos(
        round(abs(a1 * a2 + b1 * b2) / math.sqrt(a1 * a1 + b1 * b1) / math.sqrt(a2 * a2 + b2 * b2), 2)) * 180 / math.pi
    return angle


# 将属于同一文本且夹角小于阈值
def chain_link(pairs):
    chains = []
    result = []
    for pair in pairs:
        double = []
        for chain in chains:
            if have_common(pair, chain):
                double.append(chain)
        if len(double) == 0:
            chains.append(pair)
        else:
            for line in double:
                chains.remove(line)
            chains.append(unique(pair, double))
    for chain in chains:
        if len(chain) >= 1:
            result.append(chain)
    return result


def rearrange_text(chain):
    newchain = []
    index = {}
    for i, point in enumerate(chain):
        index[point[0]] = i
    new_index = sorted(index.keys())
    for i in new_index:
        newchain.append(chain[index[i]])
    return newchain


# 根据最终的文本框的宽高比判断是否为身份证号码区域
def filter_textline(chains):
    for chain in chains:
        x = []
        h = []
        for point in chain:
            x.append(point[0])
            x.append(point[0] + point[2])
            h.append(point[3])
        x_min = min(x)
        x_max = max(x)
        # 水平
        if (x_max - x_min) / np.mean(h) > 13:
            return rearrange_text(chain)

    return []


def draw_rectangle(chain, imgscr):
    x = []
    y = []
    for point in chain:
        x.append(point[0])
        x.append(point[0] + point[2])
        y.append(point[1])
        y.append(point[1] + point[3])
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    cv2.line(imgscr, (x_min, y_min), (x_min, y_max), 255, 2)
    cv2.line(imgscr, (x_min, y_min), (x_max, y_min), 255, 2)
    cv2.line(imgscr, (x_min, y_max), (x_max, y_max), 255, 2)
    cv2.line(imgscr, (x_max, y_min), (x_max, y_max), 255, 2)


def draw_rectangle1(chains, imgscr):
    for chain in chains:
        x = []
        y = []
        for point in chain:
            x.append(point[0])
            x.append(point[0] + point[2])
            y.append(point[1])
            y.append(point[1] + point[3])
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        cv2.line(imgscr, (x_min, y_min), (x_min, y_max), 255, 2)
        cv2.line(imgscr, (x_min, y_min), (x_max, y_min), 255, 2)
        cv2.line(imgscr, (x_min, y_max), (x_max, y_max), 255, 2)
        cv2.line(imgscr, (x_max, y_min), (x_max, y_max), 255, 2)


# 预处理后的图像  -->  最后的 text line
def detect_textline(img_preprocessed, imgscr):
    rects = []
    imgcopy = img_preprocessed.copy()
    # cv2.imwrite('./preprocessed_line.png', img_preprocessed)
    img, contours, hierarchy = cv2.findContours(imgcopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if rect(x, y, w, h, img_preprocessed, contours):
            rects.append([x, y, w, h])


    rects_filter = filter_rect(rects)

    # debugger
    # for (x,y,w,h) in rects_filter:
    #     cv2.rectangle(imgscr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imwrite('./{}.png'.format(time.time()), imgscr)
    # end debugger

    # group two point to pair
    for parameter in [1, 2, 2.5]:
        pairs = component_link(rects_filter, parameter)
        # link pair to chain
        chains = chain_link(pairs)
        if len(chains) > 0:
            break

    if len(chains) == 0:
        return [], ''
    # draw_rectangle1(chains,imgscr)
    textline = filter_textline(chains)
    # if len(textline)!=0:
    # for x, y, w, h in textline:
        # cv2.rectangle(imgscr, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # draw_rectangle(textline,imgscr)
    return textline, rects_filter

# 剔除太大的框或太小的框
def box_filter(contours, minsize = 64, maxsize = 3600, minlen = 6, maxlen = 40):
    boxes = get_boxes(contours)
    boxes_f = [box for box in boxes if box[2] * box[3] in range(minsize, maxsize)
               and box[2] in range(minlen, maxlen)
               and box[3] in range(minlen, maxlen)]
    return boxes_f
    
    
def box_to_box_filter(boxes, minsize = 64, maxsize = 3600, minlen = 6, maxlen = 40):
    boxes_f = [box for box in boxes if box[2] * box[3] in range(minsize, maxsize)
               and box[2] in range(minlen, maxlen)
               and box[3] in range(minlen, maxlen)]
    return boxes_f

# 将contour转换成box
def get_boxes(contours):
    boxes = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, w, h])
    return boxes


# 显示面积
def area(rects):
    square = []
    for rect1 in rects:
        x,y,w,h = rect1
        s = w*h
        square.append(s)
    return float(sum(square))/len(square)

#剔除太小的框
def del_smallbox(rects):
    box_left = []
    stop = area(rects)
    for rect in rects:
        x, y, w, h = rect
        if w*h <= stop*0.2:
            pass
        else:
            box_left.append(rect)
    return box_left