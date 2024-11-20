# 对齐车轴过程中的粗对齐，先对齐中轴的四个参数，水平上下和俯仰滚转，再旋转角度
# 使用akaze
import cv2
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

def align_to_axle():
    aligned = False
    # data中读取正确对齐轮轴应该的图像
    axle_img = cv2.imread('../data/axle_img.png')
    
    # 使用摄像头拍摄图像
    cap0 = cv2.VideoCapture(0)
    ret, img0 = cap0.read()
    
    # 使用激光测距测角确定俯仰和滚转角度两个参数
    
    # 辅助摄像头确定旋转角度这个参数
    cap1 = cv2.VideoCapture(1)
    ret, img1 = cap1.read()
    # 使用akaze确定xy这两个参数
    akaze = cv2.AKAZE_create()
    kp,desc = akaze.detectAndCompute(axle_img,None)
    points = np.array([[kp.pt[0],kp.pt[1]] for kp in kp])
    tire_center = np.mean(points,axis=0)
    # 计算相机获取图像的中心
    img_center = np.array([img0.shape[1]/2, img0.shape[0]/2])
    
    # 计算横纵轴偏移量
    offset_x = tire_center[0] - img_center[0]
    offset_y = tire_center[1] - img_center[1]
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