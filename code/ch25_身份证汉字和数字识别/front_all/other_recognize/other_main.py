
import sys
sys.path.append('../')

from other_recognize.main_cut_v2_without_cut_line_sorted_part import do_main_cut_v2_without_cut_line_sorted_part
from other_recognize.do_recognition_for_sorted_id_card import do_recognition_for_sorted_id_card

### send loaded model ###
def other_propcessing(other_img_list, model):
    ### send loaded model ###
    line_2_character_list = do_main_cut_v2_without_cut_line_sorted_part(other_img_list, model)


    results_list = []
    for line_list in line_2_character_list:
        ### send loaded model ###
        results_list.append(do_recognition_for_sorted_id_card(line_list, model))

    return results_list, line_2_character_list





