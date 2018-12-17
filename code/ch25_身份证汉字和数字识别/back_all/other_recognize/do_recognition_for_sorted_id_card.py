# -*- coding: utf-8 -*-

# caculate accuracy of model by picture
from __future__ import absolute_import
from __future__ import print_function
import os
import cv2
import numpy as np
import shutil
from keras.models import model_from_json
import csv

import time

## globale ##
# 设置切割图片的像素阈值
cutThreahold = 70
# 设置分辨率的统一值
size = 32
# 设置导入图片的尺寸和通道
img_rows, img_cols = 32, 32
# 单通道(灰度图片)-1  RGB图片-3
img_channels = 1
# 设定每次进入计算的样本batch尺寸
batch_size=50


# 2. 图像预处理
# 灰度化 - 图像增强（直方图均衡化）- 二值化 - 切割边缘 - 统一分辨率

# 2.1 灰度化
def gray_img_modified(imgscr):
    # 读入
    # imgscr = cv2.imread(filename)
    if imgscr is None:
        return 'Fault'
    # 灰度化
    newimg = cv2.cvtColor(imgscr,cv2.COLOR_RGB2GRAY)
    # 返回灰度化后的图片
    return newimg

# 2.2 图像增强 - 直方图均衡化
def enhance_img(img):
    # 增强
    newimg=cv2.equalizeHist(img)
    return newimg
     
# 2.3 二值化 自定义阈值
def binary_img(img):

    # cv.Threshold(src, dst, threshold, maxValue, thresholdType)
    # threshold_type=CV_THRESH_TRUNC:
    # 如果 src(x,y)>threshold，dst(x,y) = max_value; 否则dst(x,y) = src(x,y).     
    # 二值化 
    ret,newimg=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)  # 此种方法效果最好
    return newimg


# 2.4 判断像素值是否小于阈值，小于则返回像素点对应坐标
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
 
# 2.5 切割边缘   
# cutThreahold = 70 默认切割时使用的像素阈值，当像素值<70，认为是字符像素点，而非噪声
def cut_img(img,cutThreahold):
    
    # img.shape[0] -- 图像高度  img.shape[1] ---- 图像宽度
    # 得到每列最小像素值 ---图像横向
    width_val = np.min(img,axis=0)
    # 得到每行最小像素值  --- 图像纵向
    height_val = np.min(img,axis=1)
    
    # 获得截取部分的左边界
    left_point = np.min(indices(width_val,lambda x:x<cutThreahold))
    # 获得截取部分的右边界
    right_point = np.max(indices(width_val,lambda x:x<cutThreahold))
    # 获得截取部分的上边界
    up_point = np.max(indices(height_val,lambda x:x<cutThreahold))
    # 获得截取部分的下边界
    down_point = np.min(indices(height_val,lambda x:x<cutThreahold))
    # 将原图按照获得的字符边界位置进行裁剪
    newimg = img[down_point:up_point+1,left_point:right_point+1]
    # 返回裁剪后的图片
    return newimg

      
# 2.6 统一分辨率
# size为最终图片期望尺寸  
def standard_img(img,size):

    img_width = img.shape[1]  #图像宽度
    img_height = img.shape[0]  #图像高度
    
    # 创建一个空白的新图片，尺寸为size*size
    newimg = np.ones((size, size)) 
    newimg = newimg * 127
    
    # 计算图片最长的一边 
    scaleBig = np.max([img_height, img_width])
    # 将图片按照 size/最长边 的比例放大或缩小
    img_resize = cv2.resize(img,(int(img_width * size / scaleBig), int(img_height * size / scaleBig)))
    
    # 获得调整比例后的图片高度和宽度
    new_width = img_resize.shape[1]  #图片宽度
    new_height = img_resize.shape[0]  #图片高度
    
    # 获得调整比例后的图片内容放入size*size大小图片的起始位置
    offset_height = int(np.ceil((size - new_height) / 2)) 
    offset_width = int(np.ceil((size - new_width) / 2))
    
    # 将调整比例后的图片内容复制到空白图片
    for x in range(new_height):
        for y in range(new_width):
            newimg[x +offset_height, y +offset_width] = img_resize[x, y]
            
    # 返回预处理完成后的图片   
    return newimg

# 3.导入图片内容, 读取图片的灰度矩阵
def load_data(imgs):
    num = len(imgs)  # 保存图片总数
    # 创建4维空数组,各维度依次代表 图片数量;通道;长度;宽度
    data = np.empty((num, img_channels, img_rows, img_cols), dtype="float32")
    for i in range(num):
        img = imgs[i]  # 将图片转化为矩阵形式
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
    # 返回每张图的灰度矩阵
    data=data.reshape(data.shape[0],img_rows,img_cols,img_channels)
    return data
    
    
# 4.导入训练好的模型
def load_model():
    # 导入识别中文字符的模型（结构+权重）
    model=model_from_json(open('other_recognize/combined_model_structure.json').read())
    model.load_weights('other_recognize/combined_model_weight.h5')
    return model

# 导入中文字符和类别对应表    
def load_mappingList(scr):
    mappingList={}
    with open(scr,'r', encoding = 'utf-8') as df:
        for kv in [d.strip().split(',') for d in df]:
            mappingList[int(kv[0])] = kv[1]
    return mappingList

 
# 5. 对图片进行识别并返回结果
def predict_img(model, data, mappingList):
     
    # 对数据进行识别
    predict_class = model.predict_classes(data, batch_size=batch_size,verbose=0)
    predict_list = predict_class.tolist()
    result = [mappingList[i] for i in predict_list]
    
    # 识别后属于每个类别的概率
    proba = model.predict_proba(data,batch_size=batch_size,verbose=0).tolist()
    
    return result, proba

# 6. 识别出的地址是否和标签完全一致
def get_result(address,category,right_list,wrong_list,proba):
    
    for i in range(len(address)):
        tmp = sorted(proba[i], reverse=True)
        if address[i] == category[i]:
            right_list.append([address[i],category[i],'right', tmp[0],tmp[1],tmp[2]])
        else:
            wrong_list.append([address[i],category[i],'wrong', tmp[0],tmp[1],tmp[2]])
            
    return right_list,wrong_list

### direct use loaded model ###
def do_recognition_for_sorted_id_card(src_img_list, model):

    # start_time = time.time()
    # model = load_model()
    # print('other recog consume: {}'.format(time.time() - start_time))
    # 导入中文字符和类别对应表
    mappingList = load_mappingList("other_recognize/characterMap.txt")

    imgs = []

    for picture in src_img_list:
        # 灰度化
        img_Gray = gray_img_modified(picture)
        # 图像增强 - 直方图均衡化
        img_Enhance = enhance_img(img_Gray)
        # 二值化
        img_Binary = binary_img(img_Enhance)
        # 裁剪图像
        img_Cut = cut_img(img_Binary,cutThreahold)
        # 统一图片分辨率
        img_Standard = standard_img(img_Cut, size)
        # 将预处理后的图片存入列表
        imgs.append(img_Standard)
    # 读取识别图片的灰度矩阵和标签
    data = load_data(imgs)
    # 对图片进行识别并返回结果
    ### use loaded model to predict
    # category, proba = predict_img(model, data, mappingList)
    category = model.other_predict_class(data, mappingList)
    return ''.join(category)
