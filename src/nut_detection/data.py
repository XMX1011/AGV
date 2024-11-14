# 用于深度学习算法的数据收集、处理、标注和分割成训练集、验证集、测试集

import os
import cv2
import numpy as np
import random
import json
import yaml
from sklearn.model_selection import train_test_split

# 目的是收集一系列不同条件下的图像数据，主要有两类，
# 一类是拆完的，需要在不同光照角度等内容进行处理拍摄，并且带有一定的角度偏移
# 另一类的拆卸情况什么都有，除了拆完的情况，还有拆一半之类的，同时也要有一定的偏移和光照问题

# 数据集路径
raw_data_path = '../data/detection/raw_data'
os.makedirs(raw_data_path, exist_ok=True)
# 训练集、验证集、测试集的划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
# 三集路径
train_data_path = '../data/detection/train_data'
os.makedirs(train_data_path, exist_ok=True)
val_data_path = '../data/detection/val_data'
os.makedirs(val_data_path, exist_ok=True)
test_data_path = '../data/detection/test_data'
os.makedirs(test_data_path, exist_ok=True)

# 打开摄像头收集数据
photo_nums = 100
def collect_data():
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        # 保存到文件中，之后再标注
        # 保存的文件名为数字，后缀为.jpg
        # 按下q键退出
        # 保存路径为raw_data_path
        camera = cv2.VideoCapture(0)
        i = 0
        while i < photo_nums :
            ret, frame = camera.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imwrite(os.path.join(raw_data_path, f'{i}.jpg'), frame)
            i += 1
        camera.release()
        cv2.destroyAllWindows()


# 数据标注，标识是否拆卸完全，毕竟要做的是二分类，标识结果保存在data/detection/annotation.json文件中

def data_annotation():
    annotation = []
    for file_name in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file_name)
        img = cv2.imread(file_path)
        # 标注是否拆卸完全
        cv2.imshow('img', img)
        cv2.waitKey(0)
        is_complete = input(f'是否拆卸完全？(y/n) {file_name}').lower() == 'y'
        annotation.append({'file_name': file_name, 'is_complete': is_complete})
        # 保存标注结果
        with open(os.path.join(raw_data_path, 'annotations.json'), 'w') as f:
            json.dump(annotation, f)
        cv2.destroyAllWindows()

# 数据增强
# 旋转、缩放、裁剪、加噪声、模糊、光照变化等
def data_augmentation(img):
    img_h, img_w, _ = img.shape
    # 旋转
    angle = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    img = cv2.warpAffine(img, M, (img_w, img_h))
    # 缩放
    scale = random.uniform(0.8, 1.2)
    img = cv2.resize(img, (int(img_w*scale), int(img_h*scale)))
    # 裁剪
    x1 = random.randint(0, img_w-100)
    y1 = random.randint(0, img_h-100)
    x2 = x1 + 100
    y2 = y1 + 100
    img = img[y1:y2, x1:x2]
    # 加噪声
    img = img + np.random.normal(0, 10, img.shape)
    # 模糊
    img = cv2.blur(img, (5, 5))
    # 光照变化
    img = img * random.uniform(0.8, 1.2)
    return img

# 划分数据集,将数据集划分为训练集、验证集、测试集,划分结果保存在data/detection/data_split.json文件中
def data_split():
    with open(os.path.join(raw_data_path, 'annotations.json'), 'r') as f:
        annotations = json.load(f)
    # 划分时完全混合拆没拆完，在标注文件中更改
    train_files, val_files, test_files = [], [], []
    for annotation in annotations:
        if annotation['is_complete']:
            train_files.append(annotation['file_name'])
        else:
            val_files.append(annotation['file_name'])
    # 随机划分
    random.shuffle(train_files)
    random.shuffle(val_files)
    # 划分比例
    train_num = int(len(train_files) * train_ratio)
    val_num = int(len(val_files) * val_ratio)
    test_num = len(val_files) - val_num
    # 保存划分结果
    data_split = {'train_files': train_files[:train_num], 'val_files': val_files[:val_num], 'test_files': val_files[val_num:]}
    with open(os.path.join(raw_data_path, 'data_split.json'), 'w') as f:
        json.dump(data_split, f)
    # 复制文件到训练集、验证集、测试集目录
    for file_name in train_files[:train_num]:
        file_path = os.path.join(raw_data_path, file_name)
        img = cv2.imread(file_path)
        img = data_augmentation(img)
        cv2.imwrite(os.path.join(train_data_path, file_name), img)
    for file_name in val_files[:val_num]:
        file_path = os.path.join(raw_data_path, file_name)
        img = cv2.imread(file_path)
        img = data_augmentation(img)
        cv2.imwrite(os.path.join(val_data_path, file_name), img)
    for file_name in val_files[val_num:]:
        file_path = os.path.join(raw_data_path, file_name)
        img = cv2.imread(file_path)
        img = data_augmentation(img)
        cv2.imwrite(os.path.join(test_data_path, file_name), img)

if __name__ == '__main__':
    collect_data()
    data_annotation()
    data_augmentation()
    data_split()