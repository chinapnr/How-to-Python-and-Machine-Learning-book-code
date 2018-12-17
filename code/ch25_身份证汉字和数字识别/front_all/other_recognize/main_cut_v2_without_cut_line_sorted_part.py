# -*- coding:utf-8 -*-

#from cut_line_v4 import process_image as cut_line_process
#from prepare_cut_character_v2 import prepare_character
import sys
sys.path.append('../')

from other_recognize.cut_character_v1 import split_cut_image
from other_recognize.filter_using_cnn_prob_v1 import do_filter_using_cnn_prob_v1
import os
import cv2
import shutil

### send loaded model ###
def do_main_cut_v2_without_cut_line_sorted_part(line_img_list, model):

    all_splitted_img_list = []

    for line_index, line in enumerate(line_img_list):
        all_splitted_img_list.append(do_filter_using_cnn_prob_v1(split_cut_image(line), model))

    return all_splitted_img_list

