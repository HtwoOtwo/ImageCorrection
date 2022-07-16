# 图像缺陷检测批处理程序
包括轮廓自动识别筛选，区域分割，以及传统的缺陷检测算法

### 运行imgCut.py提取ROI，如果图像中有n个待检测区域，可以设置max_cnt = n，imgCut.py将自动摆正图像。

### 可选择执行imgCorrection.py将ROI设置为具体摆放姿态（e.g. 相同方向）。

### 执行detection.py以检测缺陷，可以选择use_label = True以去除ROI上的特定的pattern，该文件会自动去除竖直方向的栅线，用户可按需求修改。

### fileIO.py提供批处理调用功能。

### utils.py文件中提供了一些常用的图像处理函数。
