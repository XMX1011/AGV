# 使用螺母、螺栓模式匹配算法进行螺栓检测
# 用螺母模板进行检测，若检测不到螺母，则说明轮胎的螺母已完全拆除，可继续进行定位，
# 否则发出报警，提醒人工对残余螺母进行处理

# 深度学习搞个二分类模型

# 使用mobilenetv3
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvison.transforms as transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

All_nuts_removed = False

# 其实我想弄的是一堆照片然后分类，这样要自己建数据集和标注

# 读取图像
image = cv2.imread("../data/nut_image.jpg",cv2.IMREAD_GRAYSCALE)

# 去噪
image_blurred = cv2.GaussianBlur(image,(5,5),0)

# 图像标准化
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])])

input_tensor = transform(image_blurred).unsqueeze(0)

# 加载预训练模型，不过我觉得还是自己搞一套数据集自己训效果能好点，先写吧
model = mobilenet_v3_small(pretrained=True)
# 修改最后一层的输出层为二分类
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
# 自己训练的模型，暂时先放在这里
# 那么我就需要一套从收集数据到标注数据到数据增强到训练到模型的流程
model.load_state_dict(torch.load("../models/mobilenetv3_nut_detection.pth"))
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    # 预测结果
    _, predicted = torch.max(output, 1)

if predicted == 1:
    print("Nuts detected!")
    All_nuts_removed = False
else:
    print("All nuts removed.")
    All_nuts_removed = True

# 是否可视化结果
'''
if predicted.item() == 1:
    circles = cv2.HoughCircles(
        image_blurred,
        cv2.HOUGH_GRADIENT,
        1,20,
        param1=50,param2=30,
        minRadius=0,maxRadius=0
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (255, 255, 255), 2)
            cv2.circle(image, (x, y), 2, (255, 255, 255), 3)

cv2.imshow("Detected Nuts", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''