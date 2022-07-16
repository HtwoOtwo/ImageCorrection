import cv2
import numpy as np
import os
import math
from utils import *

def showImg(img):
    cv2.namedWindow('Image', 0)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 完成灰度化，二值化
def Image_Binarization(img_raw):
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # 灰度化
    means, dev = cv2.meanStdDev(img_raw[:, 1, :])
    #print(means[0])
    a = means[0]
    ret, img_process = cv2.threshold(img_gray, a[0], 255, cv2.THRESH_TOZERO)  # 二值化
    #showImg(img_process)
    #
    img_process = cv2.blur(img_process, (1, 10))
    #print(cv2.mean(img_process))
    ret, img_process = cv2.threshold(img_process, cv2.mean(img_process)[0], 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 二值化
    #showImg(img_process)

    return img_process


# 旋转函数
def rotate(img_rotate_raw, angle):
    (h, w) = img_rotate_raw.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # 计算二维旋转的仿射变换矩阵
    return cv2.warpAffine(img_rotate_raw, m, (w, h), borderValue=(0, 0, 0))



# 霍夫直线检测
def get_angle(img_hou_raw,img_cor_raw):
    sum_theta = 0
    img_canny = cv2.Canny(img_hou_raw, 200, 500, 3)
    #showImg(img_canny)
    #lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 100, minLineLength=0, maxLineGap=2000)
    i = 1

    LG = 1
    while LG < 1000:
        L = 2000
        while L > 500:
            lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 1, minLineLength=L, maxLineGap=LG)
            if lines is None:
                L = L - 50
            else:
                break
        if L < 510:
            LG = LG + 20
        else:
            break
    # print('i=')
    # print(i,L,LG)

    linenum = 0
    try:
        for line in lines:
            #print(type(line))
            x1, y1, x2, y2 = line[0]
            if x1==x2:
                theta = 90
            # print(line[0])
            # print((y2-y1)/(x2-x1))
            else:
                #t = float(y2-y1)/(x2-x1)
                t = float (x2 - x1)/(y2 - y1)
                theta = math.degrees(math.atan(t))
            #print(theta)
            if theta < 45 and theta > -45:
                sum_theta += theta
                linenum = linenum + 1
            #print(line[0][1])
            cv2.line(img_cor_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #showImg(img_cor_raw)
    except:
        print("图片位置偏移")
        return 0

    if linenum == 0:
        return 0
    angle = sum_theta / linenum
    #print(angle)
    #angle = average / np.pi * 180 - 90
    if angle > 45:
        angle = -90 + angle
    if angle < -45:
        angle = 90 + angle
    return angle


def correct(img_cor_raw):
    #img_process = Image_Binarization(img_cor_raw)
    if len(img_cor_raw.shape)==2: img_gary = img_cor_raw
    else: img_gary = cv2.cvtColor(img_cor_raw,cv2.COLOR_RGB2GRAY)
    img_process = cv2.adaptiveThreshold(img_gary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 10)
    img_process = ligature(img_process)
    #showImg(img_process)
    angle = get_angle(img_process,img_cor_raw)
    if angle == -1:
        print("No lines!!!")
        return 0
    return angle


def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('坐标值: ', x, y)
        # 在传入参数图像上画出该点
        #cv2.circle(param, (x, y), 1, (255, 255, 255), thickness=-1)
        img = param.copy()
        # 输出坐标点的像素值
        print('像素值：',param[y][x]) # 注意此处反转，(纵，横，通道)
        # 显示坐标与像素
        text = "("+str(x)+','+str(y)+')'+str(param[y][x])
        cv2.putText(img,text,(0,param.shape[0]),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),1)
        cv2.imshow('image', img)
        cv2.waitKey(0)




def search(path, name):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):
            if name in item:
                global result
                result.append(item_path)


#这里使用的Python 3
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d_SIFT.create()
    sift = cv2.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des

def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status


def rotateall(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img1 = img[2800:3500, 1800:2200, :]
    img1 = cv2.blur(img1, (5, 5))

    # means, dev = cv2.meanStdDev(img1)
    # print(means)
    img2 = img1 - 0
    img_bright_size = 3#3
    # showImg(img1)
    #
    # #得到对比度很大的图片
    img_bright = cv2.convertScaleAbs(img2, alpha=img_bright_size, beta=0)
    #showImg(img_bright)
    img2 = img_bright
    angle = correct(img_bright)
    #print(angle)
    imgrotate = rotate(img, -angle)#顺时针旋转
    return imgrotate

if __name__ == '__main__':
    file_name = r"path to your image"
    image = rotateall(file_name)
    showImg(image)
    