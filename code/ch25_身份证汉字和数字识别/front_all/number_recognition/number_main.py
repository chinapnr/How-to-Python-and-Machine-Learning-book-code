import sys
sys.path.append('../')

from number_recognition.Overall_Process_input_line import segment_number
from number_recognition.CNN_test_number_v3_modify import do_recognize

### send loaded model ###
# def number_processing(id_num_src_img):
def number_processing(id_num_src_img, model):
    character_img_list = segment_number(id_num_src_img)

    if character_img_list is None:
        return None

    ### send loaded model ###
    # return do_recognize(character_img_list), character_img_list
    return do_recognize(character_img_list, model), character_img_list
