# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:32:49 2017

@author: yi.xiong
"""

import pandas as pd 


def find_left(scr):
    img_data = pd.DataFrame(scr)
    
    columns = img_data.shape[1]  # width of the image
    start_col = int(columns * 0.4)  # can adjust !!!
    for i in range(start_col, columns):
        # 当某列中全部不为0，则返回True，有零存在，返回False
        if img_data[i].all(): 
            return img_data,i            
        
        # img_data.ix[0:end_row,i].all()
    return img_data,'error'
 
 
def find_right(img, left):
    columns = img.shape[1]  # width of the image
    start_col = int(columns * 0.55)   # can adjust!!!!!
    for i in range(start_col, left, -1):
        if img[i].all():
            return i
    return 'error'

def find_range(img,left,right):
    if right<= left:
        return 'error'
    center = int((right-left)/2)
    left = left + center - 3
    right = right - center + 2
    
    return left,right
    
def cut_image(img, left, right):
    sub_imgs = []
    columns = img.shape[1]  # width of the image
    sub_imgs.append(img[:,0:left-1,:])
    sub_imgs.append(img[:,right+1:columns-1,:])
    return sub_imgs
    
def cut_year(img):
    sub_imgs = []
    columns = img.shape[1]  # width of the image
    length =int(columns/4)
    
    for i in range(4):
        sub_imgs.append(img[:,i*length:(i+1)*length-1,:])
    
    return sub_imgs
  

def cut_date(img):
    sub_imgs = []
    columns = img.shape[1]  # width of the image
    center =int(columns/2)
    left_edge = center -3 
    right_edge = center +4
    
    left_center = int(left_edge/2)
    right_center = int(right_edge + (columns -right_edge)/2)
    
    sub_imgs.append(img[:,0:left_center,:])
    sub_imgs.append(img[:,left_center+1:left_edge,:])
    sub_imgs.append(img[:,right_edge:right_center,:])
    sub_imgs.append(img[:,right_center+1:columns-1,:])    
    
    return sub_imgs
    
    
def cut_year_date(imgs_preprocess, imgs):
    single_imgs = []
    for i in range(len(imgs_preprocess)):
        #查找边界, 切出年和日期
        img_data, left = find_left(imgs_preprocess[i])
        if left == 'error':
            return 'error'
        right = find_right(img_data, left)
        if right == 'error':
            return 'error'
            
        cut_range = find_range(img_data, left, right)
        if cut_range == 'error':
            return 'error'
        # sub_imgs[0]:year sub_imgs[1]:date
        sub_imgs = cut_image(imgs[i], cut_range[0], cut_range[1])
        
        # 分别切割年和日期得到最终结果
        year_imgs = cut_year(sub_imgs[0]) 
        date_imgs = cut_date(sub_imgs[1])
        single_imgs = single_imgs + year_imgs + date_imgs
        
    return single_imgs
        


    






















