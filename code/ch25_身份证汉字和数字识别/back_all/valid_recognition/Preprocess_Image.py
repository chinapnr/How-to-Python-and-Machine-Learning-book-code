# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:32:49 2017

@author: yi.xiong
"""

import cv2
  
        
# 1.1 灰度化
def gray_img(imgscr):
    # 灰度化
    newimg = cv2.cvtColor(imgscr,cv2.COLOR_RGB2GRAY)
    # 返回灰度化后的图片
    return newimg
    
# 1.2 图像增强 - 直方图均衡化
def enhance_img(img):
    # 增强
    newimg=cv2.equalizeHist(img)
    return newimg
     
# 1.3 二值化 自定义阈值
def binary_img(img):

    # cv.Threshold(src, dst, threshold, maxValue, thresholdType)
    # threshold_type=CV_THRESH_TRUNC:
    # 如果 src(x,y)>threshold，dst(x,y) = max_value; 否则dst(x,y) = src(x,y).     
    # 二值化 
    ret,thresh1=cv2.threshold(img,80,255,cv2.THRESH_BINARY)  

    return thresh1 




            

    






















