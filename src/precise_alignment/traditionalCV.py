# 由相机采集图像，系统对图像预处理后通过模板匹配算法快速定位车轴ROI，
# 用Hough变换对车轴外沿圆形边缘以及10个圆形螺栓轮廓边缘进行提取，
# 使用目标定位算法定位车轴相对相机位置以及螺栓相对垂直过车轴圆心的直线的夹角

import cv2
import numpy as np
from PIL import Image


axle_image_path = '../data/axle_image.jpg'
bolt_image_path = '../data/bolt_image.jpg'

def roll_angle(input_image_path,compared_image):
    aligned = False
    # 读取模板图像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # 去噪
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(image_blurred, 50, 150)
    
    # Hough变换检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        bolt_centers = []
        
        for (x, y, r) in circles[0, :]:
            cv2.circle(image, (x, y), r, (255, 255, 255), 2)
            cv2.circle(image, (x, y), 2, (255, 255, 255), 3)
            bolt_centers.append((x, y))
            
        # 确保检测到的螺栓符合数量
        if len(bolt_centers) != 10:
            print("Error: Number of detected bolts is not correct.")
            return
        
        # 计算所有螺栓的质心
        center_x = int(np.mean([center[0] for center in bolt_centers]))
        center_y = int(np.mean([center[1] for center in bolt_centers]))
        
        # 计算螺栓的角度
        angles = []
        for (x, y) in bolt_centers:
            angle = np.arctan2(y - center_y, x - center_x)*180/np.pi
            angles.append(angle)
        
        # 计算期望的角度
        expected_angle = [i*360/10 for i in range(10)]
        
        # 找到角度和期望角度的最小匹配
        min_angle_diff = float('inf')
        best_match_index = -1
        for i, angle in enumerate(angles):
            for j, expected_angle in enumerate(expected_angle):
                diff = min(abs(angle - expected_angle), 360 - abs(angle - expected_angle))
                if diff < min_angle_diff:
                    min_angle_diff = diff
                    best_match_index = j
        
        # 计算旋转方向
        if best_match_index != -1:
            target_angle = expected_angle[(best_match_index+1)%10]
            current_angle = angles[best_match_index]
            
            angle_diff = target_angle - current_angle
            if angle_diff > 180:
                direction = 'counterclockwise'
                rotation_angle = 360 - angle_diff
            else:
                direction = 'clockwise'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                rotation_angle = angle_diff
        
        # 输出旋转方向和角度
        print(f"Rotation direction: {direction}")
        print(f"Rotation angle: {rotation_angle} degrees")
        
    
    else:
        print("Error: No bolts detected.")
    return        
        
        

def precise_alignment(image_path):
    aligned = False
    # 读取车轴图像
    image = cv2.imread(axle_image_path)
    # 打开相机读取待对齐的图像
    cap0 = cv2.VideoCapture(0)
    ret_main, frame_main = cap0.read()
    cap1 = cv2.VideoCapture(1)
    ret_ass, frame_ass = cap1.read()
    if not ret_main or not ret_ass:
        print("Error: Could not read frame from camera.")
        return aligned
    # 图像预处理,相机本身拍摄的就是黑白画面
    frame_main = cv2.GaussianBlur(frame_main, (5, 5), 0)
    frame_ass = cv2.GaussianBlur(frame_ass, (5, 5), 0)
    
    # 模板匹配算法快速定位车轴ROI
    template = cv2.imread(image_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame_main, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame_main, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        roi = frame_main[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
        # cv2.imshow('ROI', roi)
    
    
    # 使用akaze特征点检测算法提取特征点来准备匹配
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(roi, None)
    
    
    
    
    # Hough变换提取车轴外沿圆形边缘以及10个圆形螺栓轮廓边缘
    # 用霍夫变换检测圆形边缘
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)
    # 用霍夫变换检测螺栓轮廓边缘
    
    
    
    # 目标定位算法定位车轴相对相机位置
    # 由第二个摄像头判断螺栓相对垂直过车轴圆心的直线的夹角
    # 检测的目标是轮胎，所以正负不会超过360/10度，因为有10个螺栓
    
    return aligned