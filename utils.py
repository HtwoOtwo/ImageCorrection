#encoding=utf-8
import cv2
import numpy as np
import math
import os
import PIL.Image as Image

class Config:
    def __init__(self):
        pass

    src = r"D:\cut_image\1.jpg"
    folder_path = r"D:\cut_image"
    minArea = 800000
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    kernel2 = np.ones((5, 5), dtype=np.uint8)
    kernel_sharpen_1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    kernel_sharpen_2 = np.array([
        [1, 1, 1],
        [1, -7, 1],
        [1, 1, 1]])
    kernel_sharpen_3 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]]) / 8.0
    resizeRate = 1

def showImg(img):
    cv2.namedWindow('Image', 0)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readImg(img_path, gray=True):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def saveImg(img, save_path):
    cv2.imencode('.jpg', img)[1].tofile(save_path)

def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('坐标值: ', x, y)
        # 在传入参数图像上画出该点
        # cv2.circle(param, (x, y), 1, (255, 255, 255), thickness=-1)
        img = param.copy()
        # 输出坐标点的像素值
        print('像素值：', param[y][x])  # 注意此处反转，(纵，横，通道)
        # 显示坐标与像素
        text = "(" + str(x) + ',' + str(y) + ')' + str(param[y][x])
        cv2.putText(img, text, (0, param.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
        cv2.imshow('image', img)
        cv2.waitKey(0)


# 得到图片像素值
def get_point_value(img):
    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', get_point, img)
    # 显示图像
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def png2jpg(path):
    for filename in os.listdir(path):  # 文件夹里不止一张图片，所以要用for循环遍历所有的图片
        if os.path.splitext(filename)[1] == '.png':  # 把path这个路径下所有的文件都读一遍，如果后缀名是png，进行下一步，即imread的读取
            img_path = path + '/' + filename
            img = cv2.imread(img_path)
            # print(img)
            # print(img_path)
            # print(filename.replace(".png", ".jpg"))
            newfilename = filename.replace(".png", ".jpg")  # 用replace函数把.png换成.jpg
            new_path = path + '/' + newfilename
            print(new_path)
            cv2.imwrite(new_path, img)

def clahe(img):
    if len(img.shape)==3:
        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        img_clahe = cv2.merge([b, g, r])
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
    return img_clahe

def imgBrightness(img, c, b):
    rows, cols = img.shape
    blank = np.zeros([rows, cols], img.dtype)
    rst = cv2.addWeighted(img, c, blank, 1-c, b)
    return rst

def getLargestContour(img):
    _, dst = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY)
    # print(dst)
    # 提取轮廓
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓面积
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return [contours[0]]

def getContourCenter(cnt):
    try:
        M = cv2.moments(cnt)
    except:
        print("ERROR: invalid contour")

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY]

def distance(p1,p2):
    X = p1[0]-p2[0]
    Y = p1[1]-p2[1]
    return math.sqrt((X ** 2) + (Y ** 2))

def ligature(img,i=4,j=5):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=i)
    img = cv2.dilate(img, kernel, iterations=j)
    return img

def rotate(img_rotate_raw, angle):
    (h, w) = img_rotate_raw.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # 计算二维旋转的仿射变换矩阵
    return cv2.warpAffine(img_rotate_raw, m, (w, h), borderValue=(0, 0, 0))

def linePts(startx, starty, endx, endy, length):
    if endx==startx: theta = np.pi/2
    else: theta = np.arctan(abs(endy - starty)/abs(endx - startx))
    if endy>starty:
        newendy = int(length*np.sin(theta)+starty)
        newendx = int(length*np.cos(theta)+startx)
    else:
        newendy = int(starty - length * np.sin(theta))
        newendx = int(length * np.cos(theta) + startx)
    pts = []
    if startx >= endx:
        for x in range(endx, startx):
            y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
            pts.append([x,y])
    else:
        for x in range(startx, endx):
            y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
            pts.append([x, y])
        for x in range(endx, newendx):
            y = int((x - endx) * (newendy - endy) / (newendx - endx)) + endy
            pts.append([x, y])
    return pts

def remove_small_objects(img, size):
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 10)
    contours, hierarch = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < size:
            cv2.drawContours(img, [contours[i]], 0, 255, -1)
    return img