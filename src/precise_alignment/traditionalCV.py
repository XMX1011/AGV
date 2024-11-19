import cv2
import numpy as np

axle_image_path = '../data/axle_image.jpg'

def precise_alignment(image_path):
    # 读取车轴图像
    image = cv2.imread(axle_image_path)
    # 打开相机读取待对齐的图像
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    