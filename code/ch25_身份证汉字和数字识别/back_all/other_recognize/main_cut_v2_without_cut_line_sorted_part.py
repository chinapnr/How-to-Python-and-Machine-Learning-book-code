# -*- coding:utf-8 -*-

import sys
sys.path.append('../')

from other_recognize.cut_character_v1 import split_cut_image
from other_recognize.filter_using_cnn_prob_v1 import do_filter_using_cnn_prob_v1
from other_recognize.prepare_cut_character_v2 import prepare_character
import os
import cv2
import shutil

def do_main_cut_v2_without_cut_line_sorted_part(line_img_list, model):

    all_splitted_img_list = []

    for line_index, line in enumerate(line_img_list):
        all_splitted_img_list.append(do_filter_using_cnn_prob_v1(split_cut_image(prepare_character(line)), model))

    return all_splitted_img_list
