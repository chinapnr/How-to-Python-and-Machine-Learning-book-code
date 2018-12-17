import numpy as np
import os

### use model_loader
from model_loader import id_card_model

from rotate_cert.FaceDetection_usedto_adjust_rotate_via_MTCNN import detectFace
# from east_part.east_segment_line import east_main
from number_recognition.number_main import number_processing
from other_recognize.other_main import other_propcessing
import cv2

import time

def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

### send models to cert_front_recog function ###
def cert_front_recog(src_img, models, debug_flag=False):

    debug_path = './{}'.format(time.time())
    if debug_flag:
        check_dir(debug_path)

    # 通过人脸旋转校正, face_count为检测到的人脸数量,若为0则代表没检测到,给予警告使用原始图片
    ### send mtcnn models to detectFace ###
    face_count, rotated_img = detectFace(src_img, models.mtcnn_models)
    if face_count < 1:
        print('warning: not find face use origin image!')
    if debug_flag:
        with open(os.path.join(debug_path, 'rotate_cert.txt'), 'w') as out:
            out.write('face_count: {}\n'.format(face_count))
        cv2.imwrite(os.path.join(debug_path, 'rotate_cert.jpg'), rotated_img)

    # EAST检测切割行并将结果排序, im为EAST输出图片,east_outpoints是EAST输出的坐标,line_cutted_imgs是行切割后图片
    ### use model loader to do east predict ###
    im, east_outpoints, line_cutted_imgs = models.east_predict(rotated_img)
    # im, east_outpoints, line_cutted_imgs = east_main(rotated_img)

    # 识别行时出现漏或多识别文字行情况,无法确定当前哪行为姓名部分无法继续进行识别,直接返回None
    if line_cutted_imgs is None:
        return None
    else:
        line_index = []
        line_content = []

        for k, v in line_cutted_imgs.items():
            row_index, column_index = k.split('_')
            line_index.append(row_index)
            line_content.append(v)

        sorted_index = np.argsort(line_index)
        sorted_line = [line_content[d] for d in sorted_index]

    if debug_flag:
        with open(os.path.join(debug_path, 'east_main_outpoint.txt'), 'w') as out:
            for line in east_outpoints:
                out.write('{}\n'.format(','.join([str(d) for d in line])))
        cv2.imwrite(os.path.join(debug_path, 'east_output_img.jpg'), im)
        debug_line_img_path = os.path.join(debug_path, 'line_img')
        check_dir(debug_line_img_path)
        # save line
        for line_findex, line in enumerate(sorted_line):
            cv2.imwrite(os.path.join(debug_line_img_path, '{}.png'.format(line_findex)), line)

    # 分为两个部分,1. 识别身份卡号, 2. 识别姓名及地址

    # 处理身份证号部分
    id_number_src_img = sorted_line[-1]
    ### send loaded number model ###
    # id_number_results, number_characters = number_processing(id_number_src_img)
    id_number_results, number_characters = number_processing(id_number_src_img, models)

    if debug_flag:
        debug_id_character_path = os.path.join(debug_path, 'number_character')
        check_dir(debug_id_character_path)
        for number_findex, ch in enumerate(number_characters):
            cv2.imwrite(os.path.join(debug_id_character_path, '{}.png'.format(number_findex)), ch)

    # 处理姓名及地址部分
    other_part_src_imgs = sorted_line[:-1]
    ### send loaded model ###
    other_result, line_characters = other_propcessing(other_part_src_imgs, models)

    if debug_flag:

        debug_other_character_path = os.path.join(debug_path, 'other')
        check_dir(debug_other_character_path)

        for folder_index, character_list in enumerate(line_characters):
            save_level_1 = os.path.join(debug_other_character_path, str(folder_index))
            check_dir(save_level_1)
            for findex, ch in enumerate(character_list):
                cv2.imwrite(os.path.join(save_level_1, '{}.png'.format(findex)), ch)

    return other_result + [str(id_number_results)]

if __name__ == '__main__':

    ### first init all model ###
    models = id_card_model()


    start = time.time()

    src_img = cv2.imread('./test.png', 1)
    recog_result = cert_front_recog(src_img, models,debug_flag=True)
    with open('./test_result.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(recog_result))

    print('consume time: {}'.format(time.time() - start))