# 进行粗对齐，对齐到轮胎架
# 考虑视觉匹配轮胎中轴为主，带可能的二维码靶标定位
# TODO: 加一个根据像素个数推断实际距离
import numpy as np
import cv2
import os
from pyzbar.pyzbar import decode

target_path = '../../data/target_img.jpg'
pivot_path = '../../data/pivot_img.jpg'
tire_img_path = '../../data/tire_img.jpg'

def detect_qr_code(image):
    # 解码 qr 码
    decoded_objects = decode(image)
    if decoded_objects:
        # 获取检测到的二维码
        qr_code = decoded_objects[0].data.decode('utf-8')
        # 获取二维码的边界框和中心点
        x, y, w, h = decoded_objects[0].rect
        center_x = x + w // 2
        center_y = y + h // 2
        
        return center_x, center_y
    else:
        return None, None
def align_to_rack():
    use_qr_code = False
    aligned = False
    # 打开相机拍摄图像
    cap = cv2.VideoCapture(0)
    image = cap.read()
    # 读取轮胎图像
    tire_img = cv2.imread(tire_img_path)
    
    # 去噪
    tire_img_blurred = cv2.GaussianBlur(tire_img, (5, 5), 0)
    
    # 边缘检测
    tire_img_edges = cv2.Canny(tire_img_blurred, 50, 150)
    
    # hough变换
    lines = cv2.HoughLines(tire_img_edges, 1, np.pi/180,threshold=50,min_line_length=50,max_line_gap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(tire_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        # 计算中轴线
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # 摄像头中心位置
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        
        if use_qr_code:
            # 检测二维码
            center_x, center_y = detect_qr_code(image)

        
        # 计算偏移量
        offset_x = mid_x - center_x
        offset_y = mid_y - center_y
        
        if abs(offset_x) < 10 or abs(offset_y) < 10:
            aligned = True
        
        # 输出建议运动方向和距离
        if offset_x > 0:
            direction_x = 'right'
        elif offset_x < 0:
            direction_x = 'left'
        else:
            direction_x = 'good,no need to move'
        
        if offset_y > 0:
            direction_y = 'up'
        elif offset_y < 0:
            direction_y = 'down'
        else:
            direction_y = 'good,no need to move'
        
        print ('Suggested direction to move: x axis is {}, y axis is {}'.format(direction_x, direction_y))
        print ('Distance to move: x axis is {}, y axis is {}'.format(abs(offset_x), abs(offset_y)))
        
    
    return aligned

# 加一个使用AKAZE的
def align_to_rack_AKAZE():
    aligned = False
    # 打开相机拍摄图像
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    
    # 初始化AKAZE
    akaze = cv2.AKAZE_create()
    kp, desc = akaze.detectAndCompute(img, None)
    
    # 计算特征值中心，作为轮胎中心
    points = np.array([kp.pt for kp in kp])
    tire_center = np.mean(points, axis=0)
    
    # 计算相机获取图像的中心
    img_center = np.array([img.shape[1]/2, img.shape[0]/2])
    
    # 计算横纵轴偏移量
    offset_x = tire_center[0] - img_center[0]
    offset_y = tire_center[1] - img_center[1]
    
    # 具体是否对齐得查看之后的像素对应的距离大小，但是这是粗对齐差不多就好
    if abs(offset_x) < 10 or abs(offset_y) < 10:
        aligned = True
    
    # 输出建议运动方向和距离
    if offset_x > 0:
        direction_x = 'right'
    elif offset_x < 0:
        direction_x = 'left'
    else:
        direction_x = 'good,no need to move'
    
    if offset_y > 0:
        direction_y = 'up'
    elif offset_y < 0:
        direction_y = 'down'
    else:
        direction_y = 'good,no need to move'
    
    print ('Suggested direction to move: x axis is {}, y axis is {}'.format(direction_x, direction_y))
    print ('Distance to move: x axis is {}, y axis is {}'.format(abs(offset_x), abs(offset_y)))
    
    return aligned