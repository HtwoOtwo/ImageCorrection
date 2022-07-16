import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
import math
from PIL import Image
from rotateandsift import *

def eaualHist(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    #opencv的直方图均衡化要基于单通道灰度图像
#     cv.namedWindow('input_image', cv.WINDOW_NORMAL)
#     cv.imshow('input_image', gray)
    dst = cv2.equalizeHist(img)                #自动调整图像对比度，把图像变得更清晰
    return dst

def cutImage(img, contours, extend=False, l=100):#l=100
    W,H,_=img.shape
    imgs = []
    if len(contours)==0:
        print("no valid contours")
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        #cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 5)
        up = y
        down = y + h
        left = x
        right = x + w
        if extend:
            if y-l>=0: up = y-l
            if y +h + l < W: down = y +h + l
            if x-l>=0:left = x-l
            if x+w+l<H: right = x+w+l

        imgs.append(img[up:down, left:right, :])  # 先用y确定高，再用x确定宽
    return imgs

def imgCutCall(img_path, max_cnt, area_range):
    '''
    如果轮廓划分不准确可以修改：ligature函数,cv2.convertScaleAbs中的alpha
    :param img_path: 图片的完整路径
    :param file_name: 图片名
    :param max_cnt: 最大的轮廓数
    :param area_range: 轮廓的面积范围
    :return: 分割后的图片s的一个列表
    '''
    #先旋转摆正
    img2 = rotateall(img_path)
    #showImg(img2)
    img = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    #print(img.shape)

    output_1 = cv2.GaussianBlur(img, (5, 5), 0)
    #output_1 = eaualHist(output_1)
    output_1 = cv2.convertScaleAbs(output_1, alpha=3, beta=0)#alpha=5

    #showImg(output_1)
    # 自适应分割
    dst = cv2.adaptiveThreshold(output_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 10)

    dst = ligature(dst)

    #_, dst = cv2.threshold(output_1, 115, 255, cv2.THRESH_BINARY)
    #showImg(dst)
    # print(dst)
    # 提取轮廓
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 计算轮廓面积
    validContours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = 0
    minArea, maxArea = area_range
    for i in contours:
        area = cv2.contourArea(i)
        #print(area)
        if area>minArea and area<maxArea:
            validContours.append(i)
            cnt += 1
        if cnt == max_cnt:
            break

        # maxArea = max(area, maxArea)
        # areas.append(area)

    return cutImage(img2, validContours, True)



if __name__ == '__main__':

    img_path = r"path to your image"
    img = imgCutCall(img_path, 1,[900000, 12000000])[0]
    showImg(img)

