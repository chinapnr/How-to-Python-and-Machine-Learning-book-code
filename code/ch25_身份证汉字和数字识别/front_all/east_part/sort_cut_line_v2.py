import numpy as np
import cv2
import os

def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

def load_region(file_name):
    rect_list = []
    with open(file_name, 'r') as reader:
        for line in reader.readlines():
            rect_list.append([int(d) for d in line[:-1].split(',')])
    return rect_list

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

def sort_rect(rect_list, over_lap_ratio=0.5):
    is_flap = False

    # filter the useless rect (currently not good idea)

    # assign only rect in id card is remain
    y_up_list = [d[1] for d in rect_list]
    y_down_list = [d[5] for d in rect_list]
    width_list = [d[2] - d[0] for d in rect_list]
    x_left_list = [d[0] for d in rect_list]

    max_width_index = np.argmax(width_list)

    id_num_y_up = y_up_list[max_width_index]

    if id_num_y_up == np.max(y_up_list):
        # the id num in the bottom not flap
        is_flap = False
    elif id_num_y_up == np.min(y_up_list):
        # the id num in the top must flap
        is_flap = True
    else:
        return None, None

    sorted_index = np.argsort(y_up_list).tolist()[::-1]

    line_rect_list = []
    before_rects_index = sorted_index.pop(0)
    cache = [before_rects_index]

    while len(sorted_index) > 0:
        current_rects_index = sorted_index.pop(0)
        before_rects_y_down = y_up_list[before_rects_index]
        before_rects_y_up = y_down_list[before_rects_index]
        current_rects_y_down = y_up_list[current_rects_index]
        current_rects_y_up = y_down_list[current_rects_index]

        if float(current_rects_y_up - before_rects_y_down) / (float(before_rects_y_up - current_rects_y_down) + 0.00001) > over_lap_ratio:
            cache.append(current_rects_index)
        else:
            cache_x_left_list = [x_left_list[d] for d in cache]
            cache_index = np.argsort(cache_x_left_list)
            cache = [cache[d] for d in cache_index]
            line_rect_list.append(cache)
            cache = [current_rects_index]
        before_rects_index = current_rects_index
    line_rect_list.append(cache)

    return line_rect_list[::-1], is_flap

# def process_sort_cut_line(src_path, txt_filename):
def process_sort_cut_line(img, rect_list):
    # rect_list = load_region(os.path.join(src_path, txt_filename))
    # rect_list = load_region(os.path.join(src_path, txt_filename))

    line_rect_list, is_flap = sort_rect(rect_list)

    if line_rect_list is None and is_flap is None:
        print('noise in file: {}'.format(f))
        return None

    return_img_dict = {}

    if is_flap:
        line_rect_list = line_rect_list[::-1]
        for i, line_list in enumerate(line_rect_list):
            for j, rect_index in enumerate(line_list):
                # img = cv2.imread(os.path.join(src_path, txt_filename.replace('.txt', '.png')), 1)
                x1, y1, x2, y2, x3, y3, x4, y4 = rect_list[rect_index]
                if i == 3 or i == len(line_rect_list) - 1:
                    start = x1 - 1 if x1 - 1 >= 0 else 0
                    end = x3 + 4 if x3 + 4 <= img.shape[1] else img.shape[1]
                    # cutted_line = img[y1: y3, x1-3: x3+5]
                elif i == 1 or i == 2:
                    continue
                else:
                    start = x1 if x1 >= 0 else 0
                    end = x3 if x3 <= img.shape[1] else img.shape[1]
                cutted_line = img[y1: y3, start: end]
                cutted_line = rotate(cutted_line, 180)
                # cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(i, j)), cutted_line)
                return_img_dict['{}_{}'.format(i, g)] = cutted_line
    else:
        for i, line_list in enumerate(line_rect_list):
            for j, rect_index in enumerate(line_list):
                # img = cv2.imread(os.path.join(src_path, txt_filename.replace('.txt', '.png')), 1)
                x1, y1, x2, y2, x3, y3, x4, y4 = rect_list[rect_index]
                if i == 3 or i == len(line_rect_list) - 1:
                    start = x1 - 1 if x1 - 1 >= 0 else 0
                    end = x3 + 4 if x3 + 4 <= img.shape[1] else img.shape[1]
                elif i == 1 or i == 2:
                    continue
                else:
                    start = x1 if x1 >= 0 else 0
                    end = x3 if x3 <= img.shape[1] else img.shape[1]
                # cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(i, j)), img[y1: y3, start: end])
                return_img_dict['{}_{}'.format(i, j)] = img[y1: y3, start: end]
    return return_img_dict
