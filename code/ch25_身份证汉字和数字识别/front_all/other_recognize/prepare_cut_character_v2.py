import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import hog

def prepare_character(src_img, threshold_value=0.05, peak_width_threshold=30):
    '''
    used to processing the image from cut_line_script_v4 , remove up, down, left, right blank edge
    make sure the images can be used in cut_character script

    main idea:
    use hog feature (detect the text texture from image), first cut left and right for text region (use the hog feature
    then dilation horizontal 10 pixels make the text part link together. find the region which width is larger than
    peak_width_threshold than cut down the text region. using the text region do hog feature again get the up and down
     cut point.

    :param src_img: source image (cut_line_script_v4 output)
    :param threshold_value: up and down point threshold
    :param peak_width_threshold: judge whether remain the peak. (used to filter the noise)
    :return:
    '''
    img = src_img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # hog feature , pixels_per_cell is related to feature size (which detected by hog), visualise (show the feature)
    fd, hog_img = hog(img,orientations=4,pixels_per_cell=(3,3), visualise=True)

    hog_img = np.asarray(hog_img, dtype='uint8')

    # normalize the feature and do binary
    cv2.normalize(hog_img, hog_img, 0, 255, cv2.NORM_MINMAX)
    res, binary_img = cv2.threshold(hog_img, 100, 255, cv2.THRESH_BINARY)

    # dilation 10 pixel let the pixel link together
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    binary_img = cv2.dilate(binary_img, element, iterations = 1)

    # start to do the splitting
    # x mapping
    x_map = np.mean(binary_img, axis=0)

    # x_map have peak (one peak is a part of the word or noise, the main text is the biggest region (sometimes will be two region))
    x_map /= np.max(x_map)

    # find word bound
    # add left and right pixel (give start and end)
    tmp_x_map = np.append(np.append([0, 0, 0], x_map), [0, 0, 0])

    x_map_candidate = np.where(tmp_x_map == 0)[0]

    # combine (get bound of each peak)
    peak_left_pts = []
    peak_right_pts = []
    temp = []
    for pt in x_map_candidate:
        if len(temp) == 0:
            temp.append(pt)
        elif pt - temp[-1] <= 3:
            temp.append(pt)
        elif pt - temp[-1] > 3:
            peak_left_pts.append(temp[-1] - 3)
            peak_right_pts.append(temp[0] - 3)
            temp = [pt]
    # add end two point
    peak_left_pts.append(temp[-1] - 3)
    peak_right_pts.append(temp[0] - 3)
    # remove the point when we first add (three zero)
    peak_left_pts.remove(peak_left_pts[-1])
    peak_right_pts.remove(peak_right_pts[0])
    # filter the noise peak
    region_width_list = []
    updated_peak_left = []
    updated_peak_right = []

    for i in range(len(peak_left_pts)):
        current_width = peak_right_pts[i] - peak_left_pts[i]
        if current_width > peak_width_threshold:
            region_width_list.append(current_width)
            updated_peak_left.append(peak_left_pts[i])
            updated_peak_right.append(peak_right_pts[i])
    # if all peak is noise return None
    if len(region_width_list) == 0:
        return None
    # add two pixel because dilation have been extend 10 pixels
    left_point = updated_peak_left[0] + 2
    right_point = updated_peak_right[-1] -2
    # left and right cut down
    left_right_cutted_img = src_img[:, left_point:right_point]

    img = cv2.cvtColor(left_right_cutted_img, cv2.COLOR_BGR2GRAY)

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    try:
        # do hog (because cutted image have possibly get error in HOG detect so use try and except
        fd, hog_img = hog(img, orientations=4, pixels_per_cell=(3, 3), visualise=True)

        hog_img = np.asarray(hog_img, dtype='uint8')

        cv2.normalize(hog_img, hog_img, 0, 255, cv2.NORM_MINMAX)
        res, binary_img = cv2.threshold(hog_img, 100, 255, cv2.THRESH_BINARY)

        y_map = np.mean(binary_img, axis=1)
        y_map /= np.max(y_map)

        # avoid error
        try:
            down_point = np.min(np.where(y_map > threshold_value)[0])
        except:
            down_point = 0
        try:
            up_point = np.max(np.where(y_map > threshold_value)[0])
        except:
            up_point = len(y_map) -1

        # adjust 1 pixel (make sure not cutted the text part)
        down_point = down_point - 1 if down_point > 0 else down_point
        up_point = up_point + 1 if up_point < src_img.shape[0] - 1 else up_point

        return src_img[down_point:up_point, left_point:right_point]
    except:
        return None

