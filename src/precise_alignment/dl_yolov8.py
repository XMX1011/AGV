# 有两个摄像头，一个在轴心主要看对齐相关的几个参数，另一个相对在下面主要看旋转角度参数
# 基于yolov8模型，要求精度

from ultralytics import YOLO
import cv2
import numpy as np
import time