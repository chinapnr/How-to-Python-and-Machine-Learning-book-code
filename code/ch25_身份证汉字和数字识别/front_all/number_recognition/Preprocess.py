# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:12:30 2017

@author: yi.xiong
"""


import cv2
import numpy as np



# 灰度世界算法
def grayWorld(imgscr):
    newimg = imgscr[::]
    # newimg = imgscr.copy()
    r = imgscr[:,:,0]
    g = imgscr[:,:,1]
    b = imgscr[:,:,2]
    avgR = np.mean(r)
    avgG = np.mean(g)
    avgB = np.mean(b)
    avgRGB = [avgR, avgG, avgB]
    grayValue = (avgR + avgG + avgB)/3  
    scaleValue = [grayValue/avg for avg in avgRGB]
    newimg[:,:,0] = scaleValue[0] * r
    newimg[:,:,1] = scaleValue[1] * g
    newimg[:,:,2] = scaleValue[2] * b
    return newimg


# 灰度化
def gray_img(imgscr):
    return cv2.cvtColor(imgscr, cv2.COLOR_RGB2GRAY)
    
    
# 图像增强
def enhance_img(imgscr):
    return cv2.equalizeHist(imgscr)
    

# 二值化
def binary_img(imgscr):
    th2 = cv2.adaptiveThreshold(imgscr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,29,15)  
    #th3 = cv2.adaptiveThreshold(imgscr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,20) 
    return th2

# 1.4 开操作  自定义结构元素
def open_img(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) # 矩形结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)) # 椭圆结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) # 十字形结构元素
    # kernel = np.uint8(np.ones((3,3)))
    
    #开运算  
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    return opened

    
# 1.5 闭操作  自定义结构元素
def close_img(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) # 矩形结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)) # 椭圆结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) # 十字形结构元素
    # kernel = np.uint8(np.ones((3,3)))
    
    #闭运算  
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed
    
# 1.6 腐蚀膨胀  自定义结构元素
def ed_img(img):
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) # 矩形结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)) # 椭圆结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5)) # 十字形结构元素
    # kernel = np.uint8(np.ones((3,3)))
    #腐蚀图像  
    eroded = cv2.erode(img,kernel) 
    #膨胀图像  
    dilated = cv2.dilate(eroded,kernel) 
    return eroded
    
    
# 预处理
def preprocess_full_img(imgscr):
    img_light = grayWorld(imgscr)
    # 灰度化
    img_gray = gray_img(img_light)
        
    img_binary = binary_img(img_gray)
    
    return img_binary


# 预处理
def preprocess_infor(imgscr):
    img_light = grayWorld(imgscr)
    # 灰度化
    img_gray = gray_img(img_light)
            
    img_binary = binary_img(img_gray)
    
    img_open = open_img(img_binary)
    
    img_ed = ed_img(img_binary)
    
    # 反色（为保证补白边）
    img = 255 - img_open
    
    return img
        
# 预处理
def preprocess(imgscr):
    img_light = grayWorld(imgscr)
    # 灰度化
    img_gray = gray_img(img_light)
    
    img_binary = binary_img(img_gray)
    
    return img_binary
        
        
        
        
        
        
        
        
        