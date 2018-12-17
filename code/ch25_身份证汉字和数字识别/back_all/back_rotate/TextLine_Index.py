# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:12:30 2017

@author: yi.xiong
"""

import cv2  
import numpy as np
import math



# 去掉嵌套的框
def filter_rect(rects):
    result = []
    for rect1 in rects:
        contain = False
        x1, y1, w1, h1 = rect1
        for rect2 in rects:
            if rect2 != rect1:
                x2, y2, w2, h2 = rect2
                if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                    contain = True
                    break
        if not contain:
            result.append(rect1)
    return result


def unique(pair, chains):
    for chain in chains:
        for point in chain:
            if point not in pair:
                pair.append(point)
    return pair


# whether two chains have at least one commone point
def have_common(pair, chain):
    p1, p2 = pair
    if p1 in chain or p2 in chain:
        return True
    else:
        return False




# 将contour转换成box
def get_boxes(contours):
    boxes = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, w, h])
    return boxes


# 显示面积
def area(rects):
    square = []
    for rect1 in rects:
        x,y,w,h = rect1
        s = w*h
        square.append(s)
    return float(sum(square))/len(square)

#剔除太小的框
def del_smallbox(rects):
    box_left = []
    stop = area(rects)
    for rect in rects:
        x, y, w, h = rect
        if w*h <= stop*0.2:
            pass
        else:
            box_left.append(rect)
    return box_left
    
    
# 计算得到两个字符框之间的距离
def get_dist(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    dist = math.sqrt(((x1 + w1/2) - (x2 + w2/2))**2 + ((y1 + h1/2) - (y2 + h2/2))**2)
    return dist
   
# 计算两对字符框之间的夹角（角度制）
def get_angle_degree(pair1, pair2):
    p1 = pair1[0]
    p2 = pair1[-1]
    p3 = pair2[0]
    p4 = pair2[-1]
    v1 = get_vector(p1, p2)
    v2 = get_vector(p3, p4)
    L1 = np.sqrt(v1.dot(v1))
    L2 = np.sqrt(v2.dot(v2))
    cos_angle = v1.dot(v2)/(L1*L2)
    angle = np.arccos(cos_angle)*360/2/np.pi
    return angle

# 根据两个字符框的中心点的得到两个字符框中心点之间连线的向量
def get_vector(p1, p2):
    x1, y1, w1, h1 = p1
    x2, y2, w2, h2 = p2
    vec = np.array([(x2-x1), (y2-y1)])
    return vec


# 从原始的框中筛选得到字符串所在的字符框   
def get_char_chain(rects):
    '''
    输入所有的外接最小矩形（框）
    根据这些框的位置和大小进行筛选
    将可能构成一些字词的框连接起来
    输出为一个列表，里面包含了全部可能的字符串，每一个字符串是一个列表，里面包含了一行（或一列）彼此相邻的字符框
    '''
    # 用pairs来存放配成对的框
    pairs = []
    for rect1 in rects:
        x1, y1, w1 ,h1 = rect1
        for rect2 in rects:
            x2, y2, w2, h2 = rect2
            if rect1 != rect2:
                # 如果两个框的宽度十分接近，并且之间的距离很近，说明这两个框可能属于同一串字符
                if abs(w1 - w2)/w1 <= 0.2 and abs(w1 - w2)/w2 <= 0.2:
                    if (min([w1, h1, w2, h2]) <= get_dist(rect1, rect2)
                    <= min([w1, h1, w2, h2])*1.7):
                        pairs.append([rect1, rect2])
    chains = []
    # 将之前配成对的框串成字符串
    for pair in pairs:
        double = []
        for chain in chains:
            # 如果pair和chain里面有共同的字符，并且它们夹角接近0或180度，说明这两个pair同属于一个字符串
            if (have_common(pair, chain) and
                (0 <= get_angle_degree(pair, chain) <= 20 or
                 160 <= get_angle_degree(pair,chain) <= 180)):
                     double.append(chain)
        # 如果没有找到合适的chain，则直接将pair加到chains里面
        if len(double) == 0:
            chains.append(pair)
        else:
            # 如果找到了可以和pair串起来的其他pair，则先将之前添加进去的chain删除，生成一个新的chain加进去
            for line in double:
                chains.remove(line)
            chains.append(unique(pair, double))
    return chains

# 获得一串字符串x值或y值的上下限
def get_chain_range(rects, axis):
    c_0_list = [rect[axis] for rect in rects]
    c_1_list = [rect[axis] + rect[axis+2] for rect in rects]
    return min(c_0_list), max(c_1_list)

# 分析两个字符串之间是否在x轴上重叠，而y轴上不重叠
def compare_chain(chain1, chain2):
    x_1_s, x_1_e = get_chain_range(chain1, 0)
    x_2_s, x_2_e = get_chain_range(chain2, 0)
    y_1_s, y_1_e = get_chain_range(chain1, 1)
    y_2_s, y_2_e = get_chain_range(chain2, 1)
    if((x_1_s <= x_2_s <= x_1_e or x_2_s <= x_1_s <= x_2_e) and 
       (y_1_e <= y_2_s)):
        return True
    else:
        return False

def get_small_text(rects):
    '''
    输入为原始的外接最小矩形（框）
    通过筛选，得到“签发机关”和“有效期限”所在字符框
    返回“签发机关”和“有效期限”所在的字符框和平均宽度
    '''
    # 首先将可能的包含字符的框通过get_char_chain筛选出来
    rects_chains = get_char_chain(rects)
    chains = list()
    # 将长度在3和4之间的字符框筛选出来
    for chain in rects_chains:
        if len(chain) >= 3 and len(chain)<= 4:
            chains.append(chain)
    issuing = list()
    valid = list()
    # 计算各个字符框串的平均宽度，生成列表
    w_list = [get_chain_width(chain) for chain in chains]
    avg_width = min(w_list)
    # 筛选剩余的字符框串，找到其中可能是“签发机关”和“有效期限”的字符串
    for ix1, chain1 in enumerate(chains):
        finish = False
        chain1_w = w_list[ix1]
        for ix2, chain2 in enumerate(chains):
            chain2_w = w_list[ix2]
            if (compare_chain(chain1, chain2) == True and
                abs(chain1_w - chain2_w) < 5):
                issuing = chain1
                valid = chain2
                avg_width = min([chain1_w, chain2_w])*0.9
                finish = True
                break
        if finish == True:
            break
        else:
            continue
    return issuing, valid, avg_width
    

# 从原始的字符框中筛选得到“居民身份证”，“中华人民共和国”所在的字符框
def get_big_text(rects):
    '''
    输入为原始的外接最小矩形（框）
    通过筛选，得到“居民身份证”， “中华人民共和国”所在字符框
    返回“居民身份证”， “中华人民共和国”所在的字符框
    '''
    # 首先将可能的包含字符的框通过get_char_chain筛选出来
    rects_chains = get_char_chain(rects)
    chains = list()
    # 将长度在3和7之间的字符框筛选出来
    for chain in rects_chains:
        if len(chain) >= 3  and len(chain) <= 7:
            chains.append(chain)
    # 计算各个字符框串的平均宽度，生成列表
    w_list = [get_chain_width(chain) for chain in chains]
    avg_width = min(w_list)
    # 找出平均宽度最大的一个字符框串
    max_index = w_list.index(max(w_list))
    prc_region = list()
    rid_region = list()
    # 如果平均宽度最长的字符框串的长度在2到5之间，说明这一串可能是“居民身份证”
    if 2 <= len(chains[max_index]) <= 5:
        rid_region = chains[max_index]
        rid_width = max(w_list)
        # 从列表里移除“居民身份证”这一字符串，剩下的字符串中平均宽度最长的一串应该就是“中华人民共和国”
        chains.remove(rid_region)
        w_list.remove(max(w_list))
        max_index = w_list.index(max(w_list))
        # “中华人民共和国”的长度应该在3到7之间，并且字符的平均宽度不会太小
        if 3 <= len(chains[max_index])<= 7 and max(w_list) >= 0.6*rid_width:
            prc_region = chains[max_index]
            prc_width = max(w_list)
        else:
            # 如果平均宽度最长的字符串不是“中华人民共和国”，则遍历剩下的字符串，看哪一条满足条件
            for chain in chains:
                if 3 <= len(chain)<=7 and get_chain_width(chain) >= 0.6*rid_width:
                    prc_region = chain
                    prc_width = get_chain_width(chain)
                    break
    else:
        for chain in chains:
            if 3 <= len(chain)<=7:
                prc_region = chain
                prc_width = get_chain_width(chain)
                break
    if prc_region != [] and rid_region != []:
        avg_width = min([prc_width*0.85,
                         rid_width*0.6])
    elif rid_region != [] and prc_region == []:
        avg_width = rid_width*0.6
    elif prc_region != [] and rid_region ==[]:
        avg_width = prc_width*0.85
    return rid_region, prc_region, avg_width
    

# 获得框的平均宽度
def get_chain_width(chain):
    chain_w = [box[2] for box in chain]
    avg_w = np.mean(chain_w)
    return avg_w

# 获得框的平均高度
def get_chain_robust_height(chain):
    chain_h = [box[3] for box in chain]
    # origin_h = chain_h.copy()
    origin_h = chain_h[::]
    try:
        chain_h.remove(max(chain_h))
        chain_h.remove(min(chain_h))
        avg_h = np.mean(chain_h)
    except:
        avg_h = np.mean(origin_h)
    return avg_h

# 获得框的平均y值
def get_chain_robust_y(chain):
    chain_y = [box[1] for box in chain]
    # origin_y = chain_y.copy()
    origin_y = chain_y[::]
    try:
        chain_y.remove(max(chain_y))
        chain_y.remove(min(chain_y))
        avg_y = int(np.mean(chain_y))
    except:
        avg_y = int(np.mean(origin_y))
    return avg_y
    


# 获得一串字符的角度，并判断身份证是水平放置还是垂直放置  
def get_text_angle(text_list):
    '''
    输入现有的字符串，例如prc_region
    判断这串字符是水平放置还是竖直放置，计算和x轴的夹角
    输出角度和是否水平放置的布尔值
    '''
    x_list = [text[0] for text in text_list]
    y_list = [text[1] for text in text_list]
    w_list = [text[2] for text in text_list]
    # 判断这一串字符的大致方向是水平的还是竖直的
    if max(x_list) + w_list[x_list.index(max(x_list))] - min(x_list) >= sum(w_list):
        hori = True
        first_char =  text_list[x_list.index(min(x_list))]
        last_char = text_list[x_list.index(max(x_list))]
    else:
        hori = False
        first_char = text_list[y_list.index(min(y_list))]
        last_char = text_list[y_list.index(max(y_list))]
    # 利用这一串字符中的第一个和最后一个计算这一串字符和水平线的夹角大小
    first_center = [first_char[0] + first_char[2]/2, first_char[1] + first_char[3]/2]
    last_center = [last_char[0] + last_char[2]/2, last_char[1] + last_char[3]/2 ]
    hdiff = last_center[1] - first_center[1]
    wdiff = last_center[0] - first_center[0]
    if wdiff == 0:
        angle = 90
    else:
        angle = np.arctan(hdiff/wdiff)*360/2/np.pi
    return angle, hori

   


# 将原图调整为正方形 
def adjustsize_img(img):
    width = img.shape[1]  #图像宽度
    height = img.shape[0]  #图像高度
    channel = img.shape[2]
    if height > width:
        size = height
    else:
        size = width
    # 创建一个空白的新图片，尺寸为size*size*size
    newimg = np.zeros((size,size,channel), np.uint8)  
    # 放入size*size大小图片的起始位置
    offset_height = int(np.ceil((size - height) / 2)) 
    offset_width = int(np.ceil((size - width) / 2))
    # 将调整比例后的图片内容复制到空白图片
    for x in range(height):
        for y in range(width):
            for i in range(channel):
               newimg[x +offset_height, y +offset_width,i] = img[x, y,i]
    # 返回预处理完成后的图片   
    return newimg


# 旋转图像
def rotate_pic(imgscr, prc_region, rid_region, hori, angle):
    '''
    输入原始图像，”居民身份证“和”中华人民共和国“的位置，是否水平的布尔值和角度
    判断“居民身份证”和“中华人民共和国”的相对位置，判断旋转的角度，并进行旋转
    输出旋转后的图像
    '''
    if hori == False and angle < 0:
        angle += 180
    else:
        pass
    # 如果角度较小，则不用专门调整，以减少图像的损失
    if abs(angle) <= 4:
        angle = 0
    # 如果角度接近90度，则将角度变为90度，从而减少图像损失
    if abs(90 - angle) <= 4:
        angle = 90
    # 使用prc_region和rid_region的x坐标和y坐标判断"居民身份证“和“中华人民共和国”的相对位置
    p_x_list = [text[0] for text in prc_region]
    p_y_list = [text[1] for text in prc_region]
    r_x_list = [text[0] for text in rid_region]
    r_y_list = [text[1] for text in rid_region]
    # 如果是垂直放置，判断”居民身份证“是否在”中华人民共和国“的左边
    try:
        if hori == False:
            if min(r_x_list) < min(p_x_list):
                angle = angle
            else:
                angle += 180
        # 如果是水平放置，判断”居民身份证“是否在”中华人民共和国“的下边
        if hori == True:
            if min(p_y_list) < min(r_y_list):
                angle = angle
            else:
                angle += 180
    except ValueError:
        pass
    # 如果angle为0，则无需旋转
    rotated = False
    if angle == 0:
        return imgscr, rotated
    else:
        rotated = True
        # 如果angle不为180，则需进行特殊处理，才能旋转
        if angle != 180:
            imgscr = adjustsize_img(imgscr)
        rows,cols = imgscr.shape[:2]
        #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        newimg = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        #第三个参数：变换后的图像大小
        newimg = cv2.warpAffine(imgscr,newimg,(cols,rows))
        return newimg, rotated


def rotate_integer_angle(imgscr, prc_region, rid_region, hori):
    '''
    输入原始图像，”居民身份证“和”中华人民共和国“的位置，是否水平的布尔值
    根据“居民身份证“和 ”中华人民共和国“的相对位置，判断旋转的角度，并进行旋转
    输出旋转后的图像
    '''
    angle = 0
    # 使用prc_region和rid_region的x坐标和y坐标判断"居民身份证“和“中华人民共和国”的相对位置
    p_x_list = [text[0] for text in prc_region]
    p_y_list = [text[1] for text in prc_region]
    r_x_list = [text[0] for text in rid_region]
    r_y_list = [text[1] for text in rid_region]
    # 如果是垂直放置，判断”居民身份证“是否在”中华人民共和国“的左边
    try:
        if hori == False:
            if min(r_x_list) < min(p_x_list):
                angle = 90
            else:
                angle = 270
        # 如果是水平放置，判断”居民身份证“是否在”中华人民共和国“的下边
        if hori == True:
            if min(p_y_list) < min(r_y_list):
                angle = 0
            else:
                angle = 180
    except ValueError:
        pass
    # 如果angle为0，则无需旋转
    rotated = False
    if angle == 0:
        return imgscr, rotated
    else:
        rotated = True
        # 如果angle不为180，则需进行特殊处理，才能旋转
        if angle != 180:
            imgscr = adjustsize_img(imgscr)
        rows,cols = imgscr.shape[:2]
        #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        newimg = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        #第三个参数：变换后的图像大小
        newimg = cv2.warpAffine(imgscr,newimg,(cols,rows))
        return newimg, rotated

 
def rotate_small_pic(ex_img, angle):
    '''
    输入原始图像和图像内字符的角度    
    根据图像内字符的角度对图像进行旋转
    返回旋转后的图像和是否旋转的布尔值
    '''
    # 如果角度较小，则不用专门调整，以减少图像的损失
    if abs(angle) <= 0.8:
        angle = 0
    # 如果angle为0，则无需旋转
    rotated = False
    if angle == 0:
        return ex_img, rotated
    else:
        rotated = True
        # 如果angle不为180，则需进行特殊处理，才能旋转
        imgscr = adjustsize_img(ex_img)
        rows,cols = imgscr.shape[:2]
        #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        newimg = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        #第三个参数：变换后的图像大小
        newimg = cv2.warpAffine(imgscr,newimg,(cols,rows))
        return newimg, rotated
    

# 计算一些外接最小矩形面积的稳健均值
def robust_area(rects):
    square = []
    for rect in rects:
        x, y, w, h = rect
        s = w*h
        square.append(s)
    square.remove(max(square))
    square.remove(min(square))
    return np.mean(square)  
    
    
# 删除一些明显太小或者太大矩形
def del_outlier_box(rects):
    box_left = []
    stop = robust_area(rects)
    for rect in rects:
        x, y, w, h = rect
        if len(rects) >= 30:
            minrate = 0.5
            maxrate = 50
        else:
            minrate = 0.2
            maxrate = 70
        if w*h <= stop * minrate or w*h >= stop * maxrate:
            pass
        else:
            box_left.append(rect)
    return box_left


# 合并较小的，不太规则的字符框，生成大的字符框
def combine_box(rects, avg_width):
    '''
    输入原始的字符框和之前得到的字符框的平均宽度
    根据平均宽度对字符框进行筛选和合并
    返回新的字符框列表
    '''
    rects_left = list()
    overlap_list = list()
    increase = 0
    # 遍历全部的框
    for rect1 in rects:
        overlap = False
        rect1_list = list()
        rect1_list.append(rect1)
        x1, y1, w1, h1 = rect1
        # 如果框的宽或高明显小于正常值，则需要进行分析
        if h1 < avg_width or w1 < avg_width:
            # 遍历全部的框
            for rect2 in rects:
                # 如果两个框不是同一个框，并且两个框十分接近并有重叠的部分
                # 并且rect2的宽度或高度也明显小于正常值，则需要进行合并
                if rect2 != rect1:
                    x2, y2, w2, h2 = rect2
                    if ((x1 <= x2 - 3 <= x1 + w1 or x1 <= x2 + w2 - 3 <= x1 + w1) and
                        (y1 <= y2 - 3 <= y1 + h1 or y1 <= y2 + h2 - 3 <= y1 +h1) and
                        (h2 < avg_width or w2 < avg_width)):
                        increase += 1
                        overlap = True
                        rect1_list.append(rect2)
                        overlap_list.append(rect2)
            # 如果之前找到的相邻或重叠的框，要将它们的边界重新划定成一个大框
            if overlap:
                rect1_x = [box[0] for box in rect1_list]
                rect1_y = [box[1] for box in rect1_list]
                rect1_xend = [box[0] + box[2] for box in rect1_list]
                rect1_yend = [box[1] + box[3] for box in rect1_list]
                new_x = min(rect1_x)
                new_y = min(rect1_y)
                new_w = max(rect1_xend) - new_x
                new_h = max(rect1_yend) - new_y
                if [new_x, new_y, new_w, new_h] not in rects:
                    rects_left.append([new_x, new_y, new_w, new_h])
            else:
                rects_left.append(rect1)
        else:
            rects_left.append(rect1)
    # 将重复添加的框删除，并且对框进行筛选
    final_rects = [rect for rect in rects_left if rect not in overlap_list]
    final_rects = filter_rect(final_rects)
    if increase == 0:
        return final_rects
    else:
        return combine_box(final_rects, avg_width)  

    
# 筛选出近似正方形的字符框
def get_square(rects):
    square_list = list()
    for rect in rects:
        x, y, w, h = rect
        if abs(w -h)/w <= 0.3 and abs(w - h)/h <= 0.3:
            square_list.append(rect)
        else:
            pass
    return square_list


def get_alter_region(img_pre, avg_width=0, mode=0):
    '''
    输入预处理后的图像
    根据所选的模式，找出图像内轮廓的外接最小矩形（框），并且进行筛选
    输出筛选后的框
    '''
    imgcopy = img_pre.copy()
    img, contours, hierarchy = cv2.findContours(imgcopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = get_boxes(contours)
    rects = del_outlier_box(rects)
    rects = filter_rect(rects)
    # 如果模式为1，则需要对一些小框进行合并
    if mode == 1:
        rects = combine_box(rects, avg_width)
    else:
        pass
    srects = get_square(rects)
    return srects



# 对字符框串根据字符框的位置从左到右进行排序
def sort_text_list(text_list):
    x_list = [text[0] for text in text_list]
    new_x_list = sorted(x_list)
    new_text_list = list()
    for x in new_x_list:
        for text in text_list:
            if text[0] == x:
                new_text_list.append(text)
                break
    return new_text_list
    
    
# 合并相邻的字符框，生成大的字符框
def combine_adjacent_box(rects):
    '''
    输入原始的字符框和之前得到的字符框的平均宽度
    根据平均宽度对字符框进行筛选和合并
    返回新的字符框列表
    '''
    rects_left = list()
    overlap_list = list()
    increase = 0
    # 遍历全部的框
    for rect1 in rects:
        overlap = False
        rect1_list = list()
        rect1_list.append(rect1)
        x1, y1, w1, h1 = rect1
        # 遍历全部的框
        for rect2 in rects:
            # 如果两个框不是同一个框，并且两个框十分接近并有重叠的部分
            # 并且rect2的宽度或高度也明显小于正常值，则需要进行合并
            if rect2 != rect1:
                x2, y2, w2, h2 = rect2
                if (x1 <= x2 - 30 <= x1 + w1 or x1 <= x2 + w2 - 30 <= x1 + w1):
                    increase += 1
                    overlap = True
                    rect1_list.append(rect2)
                    overlap_list.append(rect2)
            # 如果之前找到的相邻或重叠的框，要将它们的边界重新划定成一个大框
        if overlap:
            rect1_x = [box[0] for box in rect1_list]
            rect1_y = [box[1] for box in rect1_list]
            rect1_xend = [box[0] + box[2] for box in rect1_list]
            rect1_yend = [box[1] + box[3] for box in rect1_list]
            new_x = min(rect1_x)
            new_y = min(rect1_y)
            new_w = max(rect1_xend) - new_x
            new_h = max(rect1_yend) - new_y
            if ([new_x, new_y, new_w, new_h] not in rects and [new_x, new_y, new_w, new_h] not in rects_left):
                rects_left.append([new_x, new_y, new_w, new_h])
            else:
                increase -= len(rect1_list)
        else:
            rects_left.append(rect1)
    # 将重复添加的框删除，并且对框进行筛选
    final_rects = [rect for rect in rects_left if rect not in overlap_list]
    final_rects = filter_rect(final_rects)
    if increase == 0:
        final_rects = sort_text_list(final_rects)
        w_list = [r[2] for r in final_rects]
        l = final_rects[w_list.index(max(w_list))]
        r_list = [r for r in final_rects if (r[2] >= 20 and abs(r[3] - l[3]) <= 5 
        and (abs(r[0] - l[1]) <= 50 or abs(r[1] - l[0]) <= 50))]
        if len(r_list) >= 1 and r_list[-1][0] > l[0]:
            final_rect = r_list[-1]
        else:
            final_rect = l
        return final_rect
    else:
        return combine_adjacent_box(final_rects) 


# 合并相邻的字符框，生成大的字符框
def combine_adjacent_box_1(rects):
    '''
    输入原始的字符框和之前得到的字符框的平均宽度
    根据平均宽度对字符框进行筛选和合并
    返回新的字符框列表
    '''
    rects_left = list()
    overlap_list = list()
    increase = 0
    # 遍历全部的框
    for rect1 in rects:
        overlap = False
        rect1_list = list()
        rect1_list.append(rect1)
        x1, y1, w1, h1 = rect1
        # 遍历全部的框
        for rect2 in rects:
            # 如果两个框不是同一个框，并且两个框十分接近并有重叠的部分
            # 并且rect2的宽度或高度也明显小于正常值，则需要进行合并
            if rect2 != rect1:
                x2, y2, w2, h2 = rect2
                if (x1 <= x2 - 30 <= x1 + w1 or x1 <= x2 + w2 - 30 <= x1 + w1):
                    increase += 1
                    overlap = True
                    rect1_list.append(rect2)
                    overlap_list.append(rect2)
            # 如果之前找到的相邻或重叠的框，要将它们的边界重新划定成一个大框
        if overlap:
            rect1_x = [box[0] for box in rect1_list]
            rect1_y = [box[1] for box in rect1_list]
            rect1_xend = [box[0] + box[2] for box in rect1_list]
            rect1_yend = [box[1] + box[3] for box in rect1_list]
            new_x = min(rect1_x)
            new_y = min(rect1_y)
            new_w = max(rect1_xend) - new_x
            new_h = max(rect1_yend) - new_y
            if ([new_x, new_y, new_w, new_h] not in rects and [new_x, new_y, new_w, new_h] not in rects_left):
                rects_left.append([new_x, new_y, new_w, new_h])
            else:
                increase -= len(rect1_list)
        else:
            rects_left.append(rect1)
    # 将重复添加的框删除，并且对框进行筛选
    final_rects = [rect for rect in rects_left if rect not in overlap_list]
    final_rects = filter_rect(final_rects)
    if increase == 0:
        return final_rects
    else:
        return combine_adjacent_box_1(final_rects) 

# 删除一些很短的小框
def del_weird_region(rects, img_pre):
    left = [r for r in rects if r[3] > 0.40*img_pre.shape[0]]
    return left
    
    
def get_valid_region(img_pre):
    '''
    输入预处理后的图像
    根据所选的模式，找出图像内轮廓的外接最小矩形（框），并且进行筛选
    输出筛选后的框
    '''
    imgcopy = img_pre.copy()
    img, contours, hierarchy = cv2.findContours(imgcopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = get_boxes(contours)
    rects = del_outlier_box(rects)
    rects = filter_rect(rects)
    rects = del_weird_region(rects, img_pre)
    # 对一些小框进行合并
    final_rect = combine_adjacent_box(rects)
    return final_rect
    

def get_valid_region_1(img_pre):
    '''
    输入预处理后的图像
    根据所选的模式，找出图像内轮廓的外接最小矩形（框），并且进行筛选
    输出筛选后的框
    '''
    imgcopy = img_pre.copy()
    img, contours, hierarchy = cv2.findContours(imgcopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = get_boxes(contours)
    rects = del_outlier_box(rects)
    rects = filter_rect(rects)
    rects = del_weird_region(rects, img_pre)
    # 对一些小框进行合并
    final_rects = combine_adjacent_box_1(rects)
    return final_rects