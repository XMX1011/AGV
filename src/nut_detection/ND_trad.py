# 使用螺母、螺栓模式匹配算法进行螺栓检测
# 用螺母模板进行检测，若检测不到螺母，则说明轮胎的螺母已完全拆除，可继续进行定位，
# 否则发出报警，提醒人工对残余螺母进行处理

# 使用传统视觉匹配算法

import cv2
import yaml
import json
import numpy as np
import os
import time

fully_detached = False

# 读取图片
image = cv2.imread('../data/nut_image.jpg', cv2.IMREAD_GRAYSCALE)

# 高斯滤波
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Hough圆变换
circles = cv2.HoughCircles(image_blurred, 
                            cv2.HOUGH_GRADIENT, 
                            dp=1, mindist=20, 
                            param1=50, 
                            param2=30, 
                            minRadius=10, 
                            maxRadius=40)
# 边缘绘制
edges = cv2.Canny(image_blurred, 50, 150)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    print("检测到螺母，未拆卸完全")
    fully_detached = False
else:
    print("未检测到螺母，可能已拆卸完全")
    fully_detached = True
    
# 显示结果
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()