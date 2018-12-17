# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:29:08 2017

@author: dongyu.zhang
"""

import sys
sys.path.append('../')

import os
import cv2
import numpy as np
import shutil
import time


from back_rotate import Preprocess as pr
# import Get_Peak as gp
from back_rotate import TextLine_Index as ti


# 定义一个用来表明缺失字符框错误的类
class LackError(RuntimeError):
    def __init__(self, arg):
        self.args = arg



def overall_pre_process(imgscr):
    '''
    输入原始的图像
    经过一系列处理，找到“居民身份证”和“中华人民共和国”所在的位置
    返回“居民身份证”和“中华人民共和国”所在的位置
    '''
    rid_list = list()
    prc_list = list()
    for i in range(0,5):
        # 使用第i种预处理方式进行分析
        img_pre = eval('pr.preprocess_infor_' + str(i) + '(imgscr)')
        srects = ti.get_alter_region(img_pre)
        try:
            rid_region, prc_region, avg_width = ti.get_big_text(srects)
            rid_list.append(rid_region)
            prc_list.append(prc_region)
            # 如果找的区域达不到应有的长度，则报错，并执行二次分析
            if len(rid_region) < 5 or len(prc_region) < 7:
                raise LackError("LackError")
        except (ValueError, LackError, UnboundLocalError):
            # 使用第i种预处理方式进行二次分析
            try:
                srects = ti.get_alter_region(img_pre, avg_width, mode = 1)
                rid_region, prc_region, avg_width = ti.get_big_text(srects)
                rid_list.append(rid_region)
                prc_list.append(prc_region)
                # 如果找到区域达不到预定的长度，则报错
                if len(rid_region) < 5 or len(prc_region) < 7:
                    raise LackError("LackError")
            except (ValueError, LackError, UnboundLocalError):
                # 如果报错，则继续使用第i+1种方法寻找字符区域
                continue
            else:
                # 如果没有报错，则继续下一步运行
                break
        else:
            break
    # 如果前面找到的区域没有达到预定的长度，则要从全部找到的区域内进行筛选，找出长度最长并且满足条件的字符串作为“中华人民共和国”和“居民身份证”
    if len(rid_region) < 5 or len(prc_region) < 7:
            rid_len_list = [len(rid) for rid in rid_list]
            prc_len_list = [len(prc) for prc in prc_list]
            max_rid_index = rid_len_list.index(max(rid_len_list))
            max_prc_index = prc_len_list.index(max(prc_len_list))
            rid_region = rid_list[max_rid_index]
            prc_region = prc_list[max_prc_index]
            # 如果找到的“中华人民共和国”的字符宽度明显小于“居民身份证”的0.6，则说明找错了“中华人民共和国”，需要对剩下的部分进行筛选
            if ti.get_chain_width(rid_region)*0.6 > ti.get_chain_width(prc_region):
                # 保留满足宽度条件的字符串，找出其中最长的一串作为“中华人民共和国”
                new_prc_list = [prc for prc in prc_list if ti.get_chain_width(prc) >= ti.get_chain_width(rid_region)*0.6]
                if new_prc_list != []:
                    prc_len_list = [len(prc) for prc in new_prc_list]
                    max_prc_index = prc_len_list.index(max(prc_len_list))
                    prc_region = new_prc_list[max_prc_index]
                else:
                    pass
            else:
                pass
    return rid_region, prc_region

    


    
    
# 将身份证的区域剪切出来
def get_id_region(prc_region, rid_region, imgnew):
    '''
    输入“中华人民共和国”，“居民身份证”所在的区域以及经过旋转的图像
    根据“中华人民共和国”或者“居民身份证”的位置，确定身份证的大致范围
    将身份证剪切出来，并且计算出“中华人民共和国”，“居民身份证”在新图片中对应的区域
    返回剪切后的身份证区域图像和新的“中华人民共和国”和“居民身份证”的区域
    '''
    try:

        # 如果prc_region的长度为7，则可以使用这一区域来确定身份证的范围
        if len(prc_region) == 7:
            p_avg_h = ti.get_chain_robust_height(prc_region)
            p_avg_y = ti.get_chain_robust_y(prc_region)
            left_x = prc_region[0][0] - int(p_avg_h*5)
            top_y = p_avg_y - int(p_avg_h)
            buttom_y = p_avg_y + int(p_avg_h*8.5)
            right_x = prc_region[-1][0] + prc_region[-1][2] + int(p_avg_h*2)
        # 如果rid_region的长度为5，则可以使用这一区域来确定身份证的范围
        elif len(rid_region) == 5:
            r_avg_h = ti.get_chain_robust_height(rid_region)
            r_avg_y = ti.get_chain_robust_y(rid_region)
            left_x = rid_region[0][0] - int(r_avg_h*3.5)
            top_y = r_avg_y - int(r_avg_h*2)
            buttom_y = r_avg_y + int(r_avg_h*5.3)
            right_x = rid_region[-1][0] + rid_region[-1][2] + int(r_avg_h*1.1)
        # 其他情况下无法确定身份证的范围，只能输出原图
        else:
            print("第{0}个无法确定身份证具体位置，只能对原图进行下一步处理".format(ix))
            raise LackError("LackError")              
    except LackError:
        return prc_region, rid_region, imgnew
    else:
        # 考虑到可能出现边界点在图像外面的情况，所以需要调整边界点
        left_x = max([left_x,0])
        top_y = max([top_y,0])
        right_x = min([right_x, imgnew.shape[1]])
        buttom_y = min([buttom_y, imgnew.shape[0]])
        # 将新的id_img的范围裁剪出来，并且对prc_region和rid_region的大小进行调整
        id_img = imgnew[top_y:buttom_y, left_x:right_x, :]
        new_prc_region = [[prc[0] - left_x, prc[1] - top_y, prc[2], prc[3]] for prc in prc_region]
        new_rid_region = [[rid[0] - left_x, rid[1] - top_y, rid[2], rid[3]] for rid in rid_region]
        return new_prc_region, new_rid_region, id_img




# 将“居民身份证”以下的身份证区域剪切出来，并且调整这一部分的大小比例
def exclude_big_text_region(new_rid_region, id_img, prcmode = False):
    '''
    输入身份证区域，以及“居民身份证”或“中华人民共和国”区域，以及是否输入是否是“中华人民共和国”的布尔值
    根据“居民身份证”或“中华人民共和国”，剪切出来身份证中“中华人民共和国”的下半部分
    对剪切出的这部分的尺寸进行调整，保证输出的图像区域的大小一致
    输出调整后的区域
    '''
    # 如果不是prcmode,使用的是“居民身份证”
    if prcmode == False:
        b_list = [b[1] + b[3] for b in new_rid_region]
        # 如果是prcmode，使用的是“中华人民共和国”
    else:
        b_list = [b[1] + 3*b[3] for b in new_rid_region]
    ex_id = id_img[max(b_list):,:,:]
    rows, cols = ex_id.shape[0:2]
    points = [[0,0],[cols,0],[0,rows],[cols,rows]]
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[856,0],[0,306],[856,306]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(ex_id, M, (856,306))
    return dst



def get_text_region(ex_img):
    '''
    输入身份证上”居民身份证“以下的区域
    使用不同的预处理方法，尽可能的提取出来“签发机关”和“有效期限”所在的字符框
    返回“签发机关”和“有效期限”所在的字符框
    '''
    i_list = list()
    v_list = list()
    for i in range(0,5):
        # 使用第i种预处理方式进行分析
        img_pre = eval('pr.preprocess_infor_' + str(i) + '(ex_img)')
        cv2.imwrite('./prepare_{}.png'.format(i), img_pre)
        srects = ti.get_alter_region(img_pre)
        try:
            # 尝试寻找“签发机关”和“有效期限”所在的字符框
            issuing, valid, avg_width = ti.get_small_text(srects)
            i_list.append(issuing)
            v_list.append(valid)
            # 如果找到的“签发机关”和“有效期限”所在的字符框的长度不够，报错
            if len(issuing) < 4 or len(valid) < 4:
                raise LackError("LackError")
        except (LackError, ValueError):
            try:
                # 如果出现错误，则需要结果上一步得到平均长度，进行二次分析
                srects = ti.get_alter_region(img_pre, avg_width, mode = 1)
                i_list.append(issuing)
                v_list.append(valid)
                # 如果找到的“签发机关”和“有效期限”所在的字符框的长度不够，报错
                if len(issuing) < 4 or len(valid) < 4:
                    raise LackError("LackError")
            except (LackError, ValueError, UnboundLocalError):
                # 如果报错，则需要使用第i+1种方法再次进行分析
                continue
            else:
                # 如果没有报错，就结束循环
                break
        else:
            # 如果没有报错，就结束循环
            break
    # 如果得到的“签发机关”和“有效期限”所在的字符框的长度不够，就要从找到的全部区域中，
    # 挑出来长度最长的，进行分析
    try:
        if len(issuing) < 4 or len(valid) < 4:
            len_i = [len(i) for i in i_list]
            len_v = [len(v) for v in v_list]
            max_i_ix = len_i.index(max(len_i))
            max_v_ix = len_v.index(max(len_v))
            issuing = i_list[max_i_ix]
            valid = v_list[max_v_ix]
            try:
                # 如果挑出来的“签发机关”和“有效期限”满足位置和大小条件，则输出结果
                if (ti.compare_chain(issuing, valid) == True and
                    abs(ti.get_chain_width(issuing) - ti.get_chain_width(valid)) <= 5):
                    return issuing, valid
                else:
                    # 反之直接输出最后找到的结果吧
                    return i_list[-1], v_list[-1]
            except ValueError:
                # 报错也就直接输出最后找到的结果吧
                return i_list[-1], v_list[-1]
        else:
            return issuing, valid
    except UnboundLocalError:
        return [], []

def back_rotate(imgscr):
    try:
        rid_region, prc_region = overall_pre_process(imgscr)
    except UnboundLocalError:
        print("“居民身份证”和“中华人民共和国”定位失败")
        return None
    # 根据找到的“居民身份证”，“中华人民共和国”，计算身份证摆放的角度
    if len(prc_region) > len(rid_region):
        angle, hori = ti.get_text_angle(prc_region)
    else:
        angle, hori = ti.get_text_angle(rid_region)
    # 根据得到的角度，对图像进行旋转
    imgnew, _ = ti.rotate_pic(imgscr, prc_region, rid_region, hori, angle)

    return imgnew

if __name__ == '__main__':
    # 确定图片位置，列出图片的目录
    scr = "./src"
    # 创建存储目录和错误文件存储目录
    savedst = "./rotated"
    imgs = os.listdir(scr)
    # inforegionfail = savedst + r'/inforegionfail'
    # twolinefail = savedst + r'/twolinefail'
    # whitespacefail = savedst + r'/whitespacefail'
    # valinfofail = savedst + r'/valinfofail'
    # issinfofail = savedst + r'/issinfofail'
    if not os.path.exists(savedst):
        os.makedirs(savedst)
    # if not os.path.exists(inforegionfail):
    #     os.makedirs(inforegionfail)
    # if not os.path.exists(twolinefail):
    #     os.makedirs(twolinefail)
    # if not os.path.exists(whitespacefail):
    #     os.makedirs(whitespacefail)
    # if not os.path.exists(valinfofail):
    #     os.makedirs(valinfofail)
    # if not os.path.exists(issinfofail):
    #     os.makedirs(issinfofail)
    # timelist = list()
    # 遍历全部的图片，进行处理
    for ix, file in enumerate(imgs):
        # start = time.time()
        print(file)
        # filename = file.split(".")[0]
        # dstfolder = savedst + r'/'+ filename
        # if not os.path.exists(dstfolder):
        #     os.makedirs(dstfolder)
        shutil.copy(scr +'/' + file, './tmp.png' )
        imgscr = cv2.imread('./tmp.png')
        # 保存原图
        # origin_file = dstfolder + r'/' + file.split(".")[0] +'.png'
        # cv2.imwrite(origin_file, imgscr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # 找出“居民身份证”，“中华人民共和国”的位置
        try:
            rid_region, prc_region = overall_pre_process(imgscr)

        except UnboundLocalError:
            print("第{0}个“居民身份证”和“中华人民共和国”定位失败".format(ix))
            fail_info = inforegionfail + r'/' + file.split(".")[0] + '_fail_info.png' 
            cv2.imwrite(fail_info, imgscr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            end = time.time()
            timelist.append(end - start)
            continue
        # 根据找到的“居民身份证”，“中华人民共和国”，计算身份证摆放的角度
        if len(prc_region) > len(rid_region):
            angle, hori = ti.get_text_angle(prc_region)
        else:
            angle, hori = ti.get_text_angle(rid_region)
        # 根据得到的角度，对图像进行旋转
        imgnew, rotated = ti.rotate_pic(imgscr, prc_region, rid_region, hori, angle)

        cv2.imwrite(os.path.join(savedst, file), imgnew)
        #
        #
        # # 如果进行了旋转，则需要重新找“居民身份证”，“中华人民共和国”的位置
        # # 如果此时寻找失败，或者找的个数比以前少，则使用旋转整度数的图片进行寻找
        # # 反之直接进行后边的内容
        # if rotated == True:
        #     try:
        #         rid_region_1, prc_region_1 = overall_pre_process(imgnew)
        #     except UnboundLocalError:
        #         imgnew, rotated = ti.rotate_integer_angle(imgscr, prc_region, rid_region, angle)
        #         rid_region, prc_region = overall_pre_process(imgnew)
        #     else:
        #         if len(rid_region_1) < len(rid_region) <= 5 and len(prc_region_1) < len(prc_region) <=7:
        #             imgnew, rotated = ti.rotate_integer_angle(imgscr, prc_region, rid_region, angle)
        #             rid_region, prc_region = overall_pre_process(imgnew)
        #         else:
        #             rid_region = rid_region_1
        #             prc_region = prc_region_1
        #         del rid_region_1
        #         del prc_region_1
        # else:
        #     pass
        # # “中华人民共和国”，“居民身份证”的区域从左到右对字符框进行排序
        # prc_region = ti.sort_text_list(prc_region)
        # rid_region = ti.sort_text_list(rid_region)
        # # 将身份证区域剪切出来，并且调整“中华人民共和国”，“居民身份证”的坐标
        # new_prc_region, new_rid_region, id_img = get_id_region(prc_region, rid_region, imgnew)
        # # 如果“居民身份证”的长度大于0，则根据“居民身份证”的位置，找出“居民身份证”以下的区域，并且对其大小进行调整
        # if len(new_rid_region) > 0:
        #     ex_img = exclude_big_text_region(new_rid_region, id_img)
        # else:
        # # 反之根据“中华人民共和国”的位置找出“居民身份证”以下的区域，并且对其大小进行调整
        #     ex_img = exclude_big_text_region(new_prc_region, id_img, prcmode = True)
        # # 将“签发机关”和“有效期限”的字符框找到
        # issuing, valid = get_text_region(ex_img)
        #
        # try:
        #     # 如果没有找到“签发机关”或“有效期限”的字符框，则利用投影找到“签发机关”和“有效期限”所在行的范围
        #     if len(issuing) == 0 or len(valid) == 0:
        #         nofind = True
        #         img_pre = pr.preprocess_infor_0(ex_img)
        #         y_map = np.mean(img_pre, axis = 1)
        #         iss_range, val_range = gp.get_text_peak(y_map, min_h = 30)
        #     else:
        #         nofind = False
        #         # 计算出字符框的角度，从而判断是否需要进行旋转调整
        #         if len(issuing) >= len(valid):
        #             angle = ti.get_text_angle(issuing)[0]
        #         else:
        #             angle = ti.get_text_angle(valid)[0]
        #         # 生成旋转调整后的照片
        #         ex_img, rotated = ti.rotate_small_pic(ex_img, angle)
        #         # 如果进行了调整，则需要重新寻找“签发机关”和“有效期限”的字符框
        #         if rotated == True:
        #             issuing, valid = get_text_region(ex_img)
        #             # 如果没有找到“签发机关”或“有效期限”的字符框，则利用投影找到“签发机关”和“有效期限”所在行的范围
        #             if len(issuing) == 0 or len(valid) == 0:
        #                 nofind = True
        #                 img_pre = pr.preprocess_infor_0(ex_img)
        #                 y_map = np.mean(img_pre, axis = 1)
        #                 iss_range, val_range = gp.get_text_peak(y_map, min_h = 20)
        #             else:
        #                 # 如果找的没有问题，就根据“签发机关”或“有效期限”的字符框找到这两行所在的范围
        #                 nofind = False
        #                 iss_range, val_range = gp.get_text_range(issuing, valid)
        #         else:
        #             # 如果找的没有问题，就根据“签发机关”或“有效期限”的字符框找到这两行所在的范围
        #             nofind = False
        #             iss_range, val_range = gp.get_text_range(issuing, valid)
        #     # 根据前面找到的范围，裁剪出“签发机关”和“有效期限”所在的两行
        #     iss_line = ex_img[iss_range[0]:iss_range[1],:,:]
        #     val_line = ex_img[val_range[0]:min([val_range[1], ex_img.shape[0]]),:,:]
        #
        #     print(issuing)
        #     print(valid)
        #     cv2.imwrite('./ex_img.png', ex_img)
        #     cv2.imwrite('./iss_line.png', iss_line)
        #     cv2.imwrite('./val_line.png', val_line)
        #     # exit()
        #
        # except IndexError:
        #     print("第{0}个无法定位“签发机关”和“有效期限”".format(ix))
        #     fail_twoline = twolinefail + r'/' + file.split(".")[0] + '_fail_twoline.png'
        #     cv2.imwrite(fail_twoline, ex_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     end = time.time()
        #     timelist.append(end - start)
        #     continue
        # # 如果前面找到了“签发机关”和“有效期限”的字符框，则根据字符框，将去掉“签发机关”和“有效期限”右半部分裁剪出来
        # if nofind == False:
        #     issuing = ti.sort_text_list(issuing)
        #     valid = ti.sort_text_list(valid)
        #     if len(issuing) == 4 and len(valid) == 4:
        #        iss_ex = iss_line[:,issuing[-1][0]+2*issuing[-1][2]+10:,:]
        #        val_ex = val_line[:,valid[-1][0]+2*valid[-1][2]+10:,:]
        #     elif len(issuing) >= len(valid):
        #         n = 6 - len(issuing)
        #         iss_ex = iss_line[:,issuing[-1][0]+n*issuing[-1][2]+10:,:]
        #         val_ex = val_line[:,issuing[-1][0]+n*issuing[-1][2]+10:,:]
        #     elif len(valid) > len(issuing):
        #         n = 6 -len(valid)
        #         iss_ex = iss_line[:,valid[-1][0]+n*valid[-1][2]+10:,:]
        #         val_ex = val_line[:,valid[-1][0]+n*valid[-1][2]+10:,:]
        # else:
        #     # 如果前面没有找到了“签发机关”和“有效期限”的字符框
        #     # 则根据这两行在x轴上的投影，找出“签发机关”和“有效期限”与后文内容分割的空白所在的位置
        #     # 根据这一位置，将去掉“签发机关”和“有效期限”右半部分裁剪出来
        #     # 首先尝试使用找“签发机关”的分界点
        #     try:
        #         iss_pre = pr.preprocess_infor_1(iss_line)
        #         i_map = np.mean(iss_pre, axis = 0)
        #         i_map = max(i_map) - i_map
        #         i_space = gp.get_separated_region(i_map, min_w = 22, max_w = 60, min_h = 60)
        #     except:
        #         # 若未找到“签发机关”的分界点，再尝试寻找“有效期限”的分界点，利用“有效期限”的分界点为“签发机关”和“有效期限”分界
        #         try:
        #             val_pre = pr.preprocess_infor_1(val_line)
        #             v_map = np.mean(val_pre, axis = 0)
        #             v_map = max(v_map) - v_map
        #             v_space = gp.get_separated_region(v_map, min_w = 22, max_w = 60, min_h = 60)
        #             iss_ex = iss_line[:,v_space[1]-5:,:]
        #             val_ex = val_line[:,v_space[1]-5:,:]
        #         except:
        #             # 如果都失败，则存储错误文件，结束
        #             fail_iss = whitespacefail + r'/' + file.split(".")[0] + '_iss.png'
        #             fail_val = whitespacefail + r'/' + file.split(".")[0] + '_val.png'
        #             cv2.imwrite(fail_iss, iss_line, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #             cv2.imwrite(fail_val, val_line, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #             print("第{0}个右边部分定位失败".format(ix))
        #             end = time.time()
        #             timelist.append(end - start)
        #             continue
        #     else:
        #         try:
        #             val_pre = pr.preprocess_infor_1(val_line)
        #             v_map = np.mean(val_pre, axis = 0)
        #             v_map = max(v_map) - v_map
        #             v_space = gp.get_separated_region(v_map, min_w = 22, max_w = 60, min_h = 60)
        #             iss_ex = iss_line[:,i_space[1]-5:,:]
        #             val_ex = val_line[:,v_space[1]-5:,:]
        #         except:
        #             iss_ex = iss_line[:,i_space[1]-5:,:]
        #             val_ex = val_line[:,i_space[1]-5:,:]
        # # 对“有效期限”右半部分进行分析，找到可能的终结位置，剪切出这一行的字符内容
        # # 首先使用找轮廓的方法确定位置，一旦失败，则使用峰值投影的方法寻找位置
        # try:
        #     v_e_pre = pr.preprocess_infor_4(val_ex)
        #     rect = ti.get_valid_region(v_e_pre)
        #     val_f_index = rect[0] + rect[2]
        #     val_final = val_ex[:,:val_f_index+5,:]
        # except ValueError:
        #     for i in range(-1,5):
        #         try:
        #             if i == -1:
        #                 v_e_pre = pr.preprocess_infor_00(val_ex)
        #             else:
        #                 v_e_pre = eval('pr.preprocess_infor_' + str(i) + '(val_ex)')
        #             v_e_map = np.mean(v_e_pre, axis = 0)
        #             v_e_map = max(v_e_map) - v_e_map
        #             val_f_index = gp.get_end_index(v_e_map, min_h = 200, min_w = 30)
        #             val_final = val_ex[:,:val_f_index+5,:]
        #         except:
        #             continue
        #         else:
        #             break
        # try:
        #     val_info = dstfolder + r'/val_info.png'
        #     cv2.imwrite(val_info, val_final, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     del val_f_index
        #     del val_final
        # except:
        #     fail_val_info = valinfofail + r'/' + file.split(".")[0] + '_fail_val.png'
        #     cv2.imwrite(fail_val_info, val_ex, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     print("第{0}个“有效期限”右边部分结尾定位失败".format(ix))
        # # 对“签发机关”右半部分进行分析，找到可能的终结位置，剪切出这一行的字符内容
        # for i in range(0,5):
        #     try:
        #         i_e_pre = eval('pr.preprocess_infor_' + str(i) + '(iss_ex)')
        #         i_e_map = np.mean(i_e_pre, axis = 0)
        #         i_e_map = max(i_e_map) - i_e_map
        #         iss_f_index = gp.get_end_index(i_e_map, min_h = 130, min_w = 40, issmode = True)
        #         iss_final = iss_ex[:,:iss_f_index+5,:]
        #     except:
        #         continue
        #     else:
        #         break
        # # 对“签发机关“的字符内容部分进行处理，找出字符可能的位置，并且将字符剪切出来
        # try:
        #     min_h = 40
        #     for i in [0,-1,1,2,3,4]:
        #         try:
        #             if i == -1:
        #                 min_h = 30
        #                 i_f_pre = pr.preprocess_infor_00(iss_final)
        #             else:
        #                 i_f_pre = eval('pr.preprocess_infor_' + str(i) + '(iss_final)')
        #             i_f_map = np.mean(i_f_pre, axis = 0)
        #             peak_list = gp.get_char_index(i_f_map, min_w = 16, min_h = min_h, step = 2, times = 0, maxiter = 100)
        #             for i, p in enumerate(peak_list):
        #                 char = iss_final[:,p[0]:p[1],:]
        #                 iss_char = dstfolder + r'/iss_char_' + str(i) + r'.png'
        #                 cv2.imwrite(iss_char, char,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #                 del char
        #         except:
        #             min_h += 10
        #             if i == 4:
        #                 raise LackError("LackError")
        #             else:
        #                 continue
        #         else:
        #             break
        # except:
        #     fail_iss_info = issinfofail + r'/' + file.split(".")[0] + '_fail_iss.png'
        #     cv2.imwrite(fail_iss_info, iss_ex, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     print("第{0}个“签发机关”字符切割失败".format(ix))
        # finally:
        #     # 清除生成的变量，进入下一轮循环
        #     try:
        #         del val_ex
        #         del iss_ex
        #         del iss_final
        #         del iss_f_index
        #     except:
        #         pass
        #     end = time.time()
        #     timelist.append(end - start)
            
    # print("总时间：",sum(timelist))
    # print("处理照片总数",len(timelist))
    # print("最短时间：",min(timelist))
    # print("最长时间：",max(timelist))
    # print("平均时间：",np.mean(timelist))
    # print("中位数时间",np.median(timelist))