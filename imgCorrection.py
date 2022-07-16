import cv2
import numpy as np
from scipy import ndimage
import math
import utils
import os
from utils import *


def findCorner(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点
            if len(approx) == 4:
                docCnt = approx
                break

    for peak in docCnt:
        peak = peak[0]
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))

def correction(img,peaks,extend=False,l=10):
    '''
    :param img:
    :param peaks: 角点
    :param extend: 是否向外延伸
    :param l: 延伸的距离
    :return:
    '''
    # src: 四个顶点
    #l = 10# 向外扩张的长度
    #utils.showImg(img)
    w,h,_ = img.shape
    #print(w,h)
    peaks = np.float32(peaks)
    #print(src)
    #src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    #dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])#原图大小的四个顶点
    ###找到对应点
    tmp = [[0, 0], [0, w], [h, 0], [h, w]]
    dst = []
    for i in range(len(peaks)):
        pi = peaks[i]
        min_dis = float('inf')
        match_p = pi
        for pj in tmp:
            if distance(pi,pj)<min_dis:
                min_dis = distance(pi,pj)#距离最近的为对应点
                match_p = pj
        if extend:
            if match_p==[0,0]:
                if pi[0]-l>=0: peaks[i][0] = peaks[i][0] - l
                if pi[1]-l>=0: peaks[i][1] = peaks[i][1] - l
            if match_p==[0,w]:
                if pi[0]-l>=0: peaks[i][0] = peaks[i][0] - l
                if pi[1]+l<w: peaks[i][1] = peaks[i][1] + l
            if match_p==[h,0]:
                if pi[0]+l<h: peaks[i][0] = peaks[i][0] + l
                if pi[1]-l>=0: peaks[i][1] = peaks[i][1] - l
            if match_p==[h,w]:
                if pi[0]+l<h: peaks[i][0] = peaks[i][0] + l
                if pi[1]+l<w: peaks[i][1] = peaks[i][1] + l
        dst.append(match_p)

    dst = np.float32(dst)
    #dst = np.float32([[0, 0], [0, w], [h, 0], [h, w]])  # 原图大小的四个顶点
    #print(src)
    #print(dst)
    m = cv2.getPerspectiveTransform(peaks, dst)
    result = cv2.warpPerspective(img, m, (h, w))
    for peak in peaks:
        peak = peak.astype(int)
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    #utils.showImg(img)
    return result

def distance(p1,p2):
    X = p1[0]-p2[0]
    Y = p1[1]-p2[1]
    return math.sqrt((X ** 2) + (Y ** 2))


def peakMod(peaks, corners):
    '''
    :param peaks:
    :param min_dis: 小于该距离的角点将视为一个角点
    :return:
    '''
    if(len(peaks)<4):
        print("Error! less than 4 peaks. Try to raise up 'eps' in cv2.approxPolyDP or raise up 'min_dis' in peakMod")
    new_peaks = []
    # for i in range(len(peaks)):
    #     flag = 1
    #     for j in range(len(new_peaks)):
    #         #print(distance(peaks[i],new_peaks[j]))
    #         if(distance(peaks[i],new_peaks[j])<min_dis):
    #             flag = 0
    #             break
    #     if flag: new_peaks.append(peaks[i])
    for c in corners:
        min_dis = float("inf")
        p = peaks[0]
        for peak in peaks:
            dis = distance(peak, c)
            if min_dis > dis:
                min_dis = dis
                p = peak
        new_peaks.append(p)
    new_peaks = sorted(new_peaks, key=lambda x: (x[0], x[1]))
    return new_peaks

def correctionCall(img_path, same_ori, method = "minRect"):
    '''
    :param folder_path: 批处理文件夹
    :param same_ori: 处理的图片是否需要保证方向一致
    :param method:如果图像存在明显扭曲形变请选择 method = “PolyDP”
    :return:
    '''
    ###############
    # 受cv2.threshold影响较大
    ###############

    img = utils.readImg(img_path)
    w, h = img.shape
    img2 = utils.readImg(img_path, False)
    img = cv2.convertScaleAbs(img, alpha=3, beta=0)  # alpha=5
    #utils.showImg(img)
    # _, dst = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY)#155
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 10)
    #showImg(dst)
    #dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 10)
    # 闭运算，把断线补上
    dst = cv2.erode(dst, utils.Config.kernel1, iterations=1)
    dst = cv2.dilate(dst, utils.Config.kernel2, iterations=1)
    #utils.showImg(dst)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

    docCnt = None
    # print(len(contours))
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序

    # 最大轮廓
    maxContour = contours[0]
    # cv2.drawContours(img2, contours, 0, (255, 0, 0), 1)
    # showImg(img2)
    if method == "minRect":
        rect = cv2.minAreaRect(maxContour)
        cx, cy = rect[0]
        box = cv2.boxPoints(rect)
        peaks = np.int0(box)
    else:
        ##多边形拟合的方法
        peri = cv2.arcLength(maxContour, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(maxContour, 0.06 * peri, True)  # 轮廓多边形拟合
        docCnt = approx
        # print(len(docCnt))
        # cv2.drawContours(img, maxContour, -1, (0, 0, 255), 5)
        #utils.showImg(img)
        # 找到四个顶点
        peaks = []
        for peak in docCnt:
            peak = peak[0]
            peaks.append(peak)
            print(peak)
            #cv2.circle(img2, tuple(peak), 10, (255, 0, 0))
        showImg(img2)
        ##确保是四个顶点，通过调节min_dis
        corners = [[0, 0], [0, w], [h, 0], [h, w]]
        peaks = peakMod(peaks, corners)  # 四个顶点

    c_img = correction(img2, peaks, True)
    #utils.showImg(c_img)
    c_img = rotate(c_img, same_ori)
    return c_img

def rotate(img, same_ori = False):
    '''

    :param img:
    :param same_ori: 保证图像方向统一
    :return:
    '''
    w,h,_ = img.shape
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if same_ori:
        bin_img = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 5)
        stripes = [bin_img[:,40:70],bin_img[40:70,:],bin_img[:,max(h-70,0):max(h-40,0)],bin_img[max(w-70,0):max(w-40,0),:]]
        label = 0
        for i in range(len(stripes)):
            if(check(stripes[i])):
                label = i
        print("label: ",label)
        if label == 0:
            #img = np.rot90(img, 3)#270
            img = cv2.flip(cv2.transpose(img), 0)
        if label == 1:
            #img = np.rot90(img, 2)#180
            img = cv2.flip(img, -1)
        if label == 2:
            #img = np.rot90(img)#90
            img = cv2.flip(cv2.transpose(img), 1)
    else:
        if w>h:
            img = cv2.flip(cv2.transpose(img), 0)
    return img

def check(img):
    img = cv2.erode(img, utils.Config.kernel1, iterations=5)
    img = cv2.dilate(img, utils.Config.kernel2, iterations=5)
    #utils.showImg(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
    cnt = 0
    for obj in contours:
        area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
        #cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
        perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
        CornerNum = len(approx)  # 轮廓角点的数量
        #print(CornerNum,end=" ")
        x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度

        # 轮廓对象分类
        if CornerNum == 3:
            objType = "triangle"
        elif CornerNum == 4:
            cnt = cnt+1
            objType = "Rectangle"
        elif CornerNum > 4:
            objType = "Circle"
        else:
            objType = "N"
    print()
    if cnt>=2: return True
    else: return False


if __name__ == '__main__':
    folder_path = r"path to your image"
    #folder_path = r"D:\testImg"
    img = correctionCall(folder_path, same_ori=False,method='DP')
    showImg(img)
