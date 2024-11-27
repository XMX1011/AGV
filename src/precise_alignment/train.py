import os
import cv2
import numpy as np
import yaml
import json
import torch
import ultralytics.yolov8 as yolov8
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import tensorrt as trt

# 使用yolov8训练模型，进行对车轴和螺栓的精确对齐
# 对车轴的俯仰横滚左右旋转角度、横纵距离、轮胎旋转角度

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 读取训练数据集
train_data_path = config['train_data_path']
train_data = []
for file in os.listdir(train_data_path):
    if file.endswith('.jpg'):
        train_data.append(os.path.join(train_data_path, file))

# 训练yolov8模型


# 导出onnx模型
dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(model, dummy_input, "precise_alignment.onnx", verbose=True)

# 导出tensorrt模型

model_trt = torch.jit.load(os.path.join('models', 'nut_detection.pth'))
model_trt = model_trt.to(device)
model_trt.eval()
x = torch.randn(1, 3, 640, 640).to(device)
model_trt_path = os.path.join('models', 'nut_detection_trt.pth')
torch.save(model_trt.state_dict(), model_trt_path)
model_trt.to(device)
model_trt.eval()
model_trt = torch.jit.trace(model_trt, x)
model_trt.save(model_trt_path)