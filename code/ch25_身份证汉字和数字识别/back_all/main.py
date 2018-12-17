import numpy as np
import os

### use model loader
from model_loader import id_card_model

from back_rotate.Back_rotate import back_rotate
# from east_part.east_segment_line import east_main
from other_recognize.other_main import other_propcessing
from valid_recognition.valid_main import valid_processing

import cv2

import time

def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

def cert_back_recog(src_img, models, debug_flag=False):

    debug_path = './{}'.format(time.time())
    if debug_flag:
        check_dir(debug_path)

    # 旋转校正身份证背面
    rotated_img = back_rotate(src_img)
    if rotated_img is None:
        print('error in back_rotate')
        return None
    if debug_flag:
        cv2.imwrite(os.path.join(debug_path, 'rotate_back.jpg'), rotated_img)

    # EAST检测结果(带框图, 坐标, 切割后每一行)
    ### use model loader to do east predict ###
    im, east_outpoints, line_cutted_imgs = models.east_predict(rotated_img)
    # im, east_outpoints, line_cutted_imgs = east_main(rotated_img)

    if line_cutted_imgs is None:
        # 识别行数超过3行(无法确定多哪行或少哪行,无法继续进行识别)
        print('error in east locate')
        return None

    if debug_flag:
        with open(os.path.join(debug_path, 'east_main_outpoint.txt'), 'w') as out:
            for line in east_outpoints:
                out.write('{}\n'.format(','.join([str(d) for d in line])))
        cv2.imwrite(os.path.join(debug_path, 'east_output_img.jpg'), im)
        debug_line_img_path = os.path.join(debug_path, 'line_img')
        check_dir(debug_line_img_path)
        # save line
        for line_findex, line in enumerate(line_cutted_imgs):
            cv2.imwrite(os.path.join(debug_line_img_path, '{}.png'.format(line_findex)), line)

    # 分为两个部分1. 签发机关识别, 2. 有效日期识别
    # 签发机关部分
    iss_result, iss_characters = other_propcessing([line_cutted_imgs[0]], models)
    if debug_flag:
        debug_iss_path = os.path.join(debug_path, 'iss')
        check_dir(debug_iss_path)
        for iss_ch_findex, ch in enumerate(iss_characters[0]):
            cv2.imwrite(os.path.join(debug_iss_path, '{}.png'.format(iss_ch_findex)), ch)

    # 有效日期部分
    valid_result, valid_characters = valid_processing(line_cutted_imgs[1], models)
    if debug_flag:
        debug_val_path = os.path.join(debug_path, 'val')
        check_dir(debug_val_path)
        for val_ch_findex, ch in enumerate(valid_characters):
            cv2.imwrite(os.path.join(debug_val_path, '{}.png'.format(val_ch_findex)), ch)

    return [iss_result[0], valid_result]

if __name__ == '__main__':

    ### first init all model ###
    models = id_card_model()

    start_time = time.time()

    src_img = cv2.imread('./test.png', 1)
    recog_result = cert_back_recog(src_img, models, debug_flag=True)
    with open('test_result.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(recog_result))

    print('consume time: {}'.format(time.time() - start_time))