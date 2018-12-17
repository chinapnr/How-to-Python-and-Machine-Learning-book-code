
import sys
sys.path.append('../')

from valid_recognition.WholeProcess_valid import do_wholeprocess_valid
from valid_recognition.CNN_test_date_modify import do_CNN_test_date_modify

### send loaded model ###
def valid_processing(valid_line, model):
    valid_character = do_wholeprocess_valid(valid_line)

    ### send loaded model ###
    return do_CNN_test_date_modify(valid_character, model), valid_character
