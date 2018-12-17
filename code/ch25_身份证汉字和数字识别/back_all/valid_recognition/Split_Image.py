# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:32:49 2017

@author: yi.xiong
"""

import pandas as pd 

   

#########################################################
# abnormal     
def find_range_abnormal(img_data):    
    columns = img_data.shape[1]  # width of the image
    start = int(columns * 0.7)
    for i in range(start, 0, -1):
        if img_data[i].all():
            return i
    return 'error'
            
    
def cut_image_abnormal(img, edge):
    return [img[:,0:edge,:]]

# 只返回xxxx.xx.xx
def cut_abnormal(img_preprocess, imgscr):
    img_data = pd.DataFrame(img_preprocess)
    edge = find_range_abnormal(img_data)
    if edge != 'error':
        sub_img = cut_image_abnormal(imgscr, edge)
        return sub_img
    else:
        return 'error'


##########################################################    
    
##########################################################
# regular    
def find_center_regular(img):
    img_data = pd.DataFrame(img)
    
    columns = img_data.shape[1]  # width of the image
    
    if img_data[int(columns/2)].all() == False:
        return img_data, int(columns/2)  

    find_c = False
    find_r = False
    left_c = int(columns/2)
    right_c = int(columns/2)
    while not find_c and not find_r:
        if img_data[left_c].all():  # if all zeros, then return True
            left_c = left_c -1
        else:
            find_c = True
        if img_data[right_c].all():
            right_c = right_c + 1
        else:
            find_r = True
     
    if int(columns/2) - left_c >= right_c - int(columns/2):
        return img_data, right_c
    else:
        return img_data, left_c


def find_range_regular(img,center):
    rows = img.shape[0]  # height of the image
    for i in range(rows):
        # find start point
        if img.at[i,center] == 0:
            # find left and right point
            left = find_left_regular(i, center, img)
            right = find_right_regular(i,center, img)
            if left != 'error' and right != 'error':
                return min(left), max(right)
            else:
                return 'error', 'error'
                  
            
def find_left_regular(start_row,start_column, img):
    rows = img.shape[0]  # height of the image
    left = []
    for i in range(start_row, rows):
        for j in range(start_column, 0, -1):
            if img.at[i,j] == 255:
                left.append(j+1)
                break
    if len(left) == 0:
        return 'error'
    else:
        return left
 
 
def find_right_regular(start_row,start_column, img):
    rows = img.shape[0]  # height of the image
    columns = img.shape[1]  # width of the image
    right = []
    for i in range(start_row, rows):
        for j in range(start_column, columns):
            if img.at[i,j] == 255:
                right.append(j-1)
                break    
    if len(right) == 0:
        return 'error'
    else:
        return right

    
def cut_image_regular(img, left, right):
    sub_imgs = []
    columns = img.shape[1]  # width of the image
    sub_imgs.append(img[:,0:left-1,:])
    #sub_imgs.append(img[:,left:right,:])
    sub_imgs.append(img[:,right+1:columns-1,:])
    return sub_imgs    
    
# 返回 xxxx.xx.xx 和  xxxx.xx.xx    
def cut_regular(img_preprocess, imgscr):
    sub_imgs =[]
    img_data, center = find_center_regular(img_preprocess)
    left, right = find_range_regular(img_data, center)
    if left != 'error' and right != 'error':
        sub_imgs = cut_image_regular(imgscr, left, right)
        return sub_imgs
    else:
        return 'error'





        
        

    






















