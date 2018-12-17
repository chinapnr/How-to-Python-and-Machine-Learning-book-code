# -*- coding:utf-8 -*-

'''

切割后形成分支，待网络进行过滤

'''

import os
import cv2
import numpy as np
from skimage import filters

def standard_image_height(src_img, target_height=30):

    img = src_img.copy()

    img_rows, img_cols = img.shape[:2]

    resize_ratio = float(target_height) / float(img_rows)

    img = cv2.resize(img, (int(img_cols * resize_ratio), target_height))

    return img, resize_ratio

def find_peak_point(img, point_count=15, combine_count = 3):

    # start to do the splitting
    x_map = np.mean(img, axis=0)

    # add three pixel before and after the mapping values
    x_map = np.append(np.append([0, 0, 0], x_map), [0, 0, 0])

    map_max = np.max(x_map)
    x_map /= map_max

    local_minimum_list = []

    for i in range(len(x_map)):
        if i - point_count <= 0:
            left_lim = 0
        else:
            left_lim = i - point_count
        if i + point_count >= len(x_map):
            right_lim = len(x_map) - 1
        else:
            right_lim = i + point_count

        if x_map[i] == np.min(x_map[left_lim: right_lim]):
            local_minimum_list.append(i)

    # starting combine
    cut_points = []
    temp = []
    for pt in local_minimum_list:
        if len(temp) == 0:
            temp.append(pt)
        elif pt - temp[-1] <= combine_count:
            temp.append(pt)
        elif pt - temp[-1] > combine_count:
            cut_points.append(temp[int(len(temp) / 2)])
            temp = [pt]

    # recovery the offset
    cut_points = [d - 3 for d in cut_points]

    # make sure the first element is bigger than one
    if cut_points[0] < 0:
        cut_points[0] = 0

    # add last one
    cut_points.append(img.shape[1] - 1)

    # return result cut point
    return cut_points

def local_minimum_cut(src_img, judge_value=0.5):
    '''
    main idea find all minimum find the most lower and center point
    '''

    img = src_img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # invert
    img = 255 - img

    # binary
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    local_x_map = np.mean(img, axis=0)
    local_x_map /= np.max(local_x_map)

    candidate_point = find_peak_point(img, point_count=5)

    img_center = int(img.shape[1] / 2.)
    center_distance = [np.abs(d - img_center) for d in candidate_point]
    distance_index = np.argsort(center_distance)

    first_local = candidate_point[distance_index[0]]
    cut_pt = first_local

    if cut_pt < 0.2 * src_img.shape[1] or cut_pt > 0.8 * src_img.shape[1]:
        return None

    return [src_img[:, :cut_pt], src_img[:, cut_pt: ]]

def get_single_width(src_img, threshold_value=30):
    img = src_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    x_map = np.mean(img, axis=0)
    right_point = np.max(np.where(x_map > threshold_value)[0])
    left_point = np.min(np.where(x_map > threshold_value)[0])
    return right_point - left_point

def get_real_widths(imgs, threshold=30):
    widths = []

    for src_img in imgs:
        widths.append(get_single_width(src_img, threshold_value=threshold))

    return widths

def split_cut_image(src_img):
    global index

    # setting
    threshold_value = 30

    img, first_resize_ratio = standard_image_height(src_img)
    bgr_img = img.copy()

    # convert to LAB mode, because the illumination is unbalance. need to adjust the illumination
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    cl = clahe.apply(l)

    lab_img = cv2.merge((cl, a, b))

    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    # covert to gray scale
    img_back = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert
    img = 255 - img_back.copy()

    # normalize
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # binary
    val = filters.threshold_otsu(img)
    ret, img = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)

    # x, y mapping
    x_map = np.mean(img, axis=0)
    y_map = np.mean(img, axis=1)
    # find word bound
    right_point = np.max(np.where(x_map > threshold_value)[0])
    left_point = np.min(np.where(x_map > threshold_value)[0])
    down_point = np.min(np.where(y_map > threshold_value)[0])
    up_point = np.max(np.where(y_map > threshold_value)[0])

    img = img_back[down_point: up_point, left_point: right_point]
    bgr_img = bgr_img[down_point: up_point, left_point: right_point]

    img = cv2.equalizeHist(img)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    val = filters.threshold_otsu(img)
    ret, img = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)

    img, second_resize_ratio = standard_image_height(img)
    bgr_img, second_resize_ratio = standard_image_height(bgr_img)

    # invert image
    img = 255 - img

    cut_points = find_peak_point(img)

    origin_sub_imgs = []

    for i in range(len(cut_points) - 1):
        start_pt = cut_points[i]
        end_pt = cut_points[i + 1]
        origin_sub_imgs.append(bgr_img[:, start_pt: end_pt])

    # do combine use median to judge
    real_img_widths = get_real_widths(origin_sub_imgs)
    sub_img_widths = real_img_widths[:]
    sub_img_widths = sorted(sub_img_widths)
    character_width = sub_img_widths[int(len(sub_img_widths) / 2)]

    if character_width < 30:
        character_width = 30

    # list with branch each is a tuple one element in tuple or two element in tuple
    final_return_list_branch = []
    final_return_list_widths = []

    cache = None
    cache_list = []
    current_pt = 0
    while current_pt < len(origin_sub_imgs):

        if not cache is None:
            new_img = np.hstack((cache, origin_sub_imgs[current_pt]))
            new_width = get_single_width(new_img)
            if np.abs(new_width - character_width) <= 0.2 * character_width or new_width - character_width <= - 0.2 * character_width:
                cache = new_img
                cache_list.append(origin_sub_imgs[current_pt])
            else:
                if len(cache_list) > 0:
                    final_return_list_branch.append([cache_list, [cache]])
                else:
                    final_return_list_branch.append([[cache]])
                final_return_list_widths.append(0)
                final_return_list_branch.append([[origin_sub_imgs[current_pt]]])
                final_return_list_widths.append(real_img_widths[current_pt])
                cache = None
                cache_list = []
        elif np.abs(real_img_widths[current_pt] - character_width) <= 0.2 * character_width or real_img_widths[current_pt] - character_width > 0.2 * character_width:
            final_return_list_branch.append([[origin_sub_imgs[current_pt]]])
            final_return_list_widths.append(real_img_widths[current_pt])
        elif np.abs(real_img_widths[current_pt] - character_width) > 0.2 * character_width and cache is None:
            cache = origin_sub_imgs[current_pt]
            cache_list.append(origin_sub_imgs[current_pt])

        current_pt += 1
    if not cache is None:
        if len(cache_list) > 0:
            final_return_list_branch.append([cache_list, [cache]])
        else:
            final_return_list_branch.append([[cache]])
        final_return_list_widths.append(0)

    # multi_character processing
    process_multi_character = []
    for i, node in enumerate(final_return_list_branch):

        if len(node) != 1:
            process_multi_character.append(node)
        else:
            current_cut_img = node[0][0]
            # print(current_cut_img.shape[1])
            if final_return_list_widths[i] >= character_width:
                multi_part = local_minimum_cut(current_cut_img)
                if not multi_part is None:
                    process_multi_character.append([[current_cut_img], multi_part])
                else:
                    process_multi_character.append([[current_cut_img]])
            else:
                process_multi_character.append([[current_cut_img]])

    return process_multi_character
