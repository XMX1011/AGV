# 拍摄的原图
raw_data_path = '../data/raw/'
# 原图处理后
processed_data_path = '../data/processed/'
# 标注文件
annotation_path = '../data/annotation/'
# 训练集
train_data_path = '../data/processed/train/'
# 测试集
test_data_path = '../data/processed/test/'

import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
random.seed(42)
num_of_pic = 1000

def capture_data(raw_data_path):
    """
    捕获原始数据,打开摄像头,按下键盘输入'q'退出，按下's'保存图片
    """
    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_of_pic:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(os.path.join(raw_data_path, 'pic{}.jpg'.format(count)), frame)
            count += 1
    cap.release()
    cv2.destroyAllWindows()
    


def preprocess_data(raw_data_path, processed_data_path):
    """
    预处理数据,将原图进行resize,归一化等处理,并保存到processed_data_path目录下
    Args:
        raw_data_path (_type_): _description_
        processed_data_path (_type_): _description_
    """
    
    for filename in os.listdir(raw_data_path):
        img = cv2.imread(os.path.join(raw_data_path, filename))
        aumented_img = datagen.random_transform(img)
        cv2.imwrite(os.path.join(processed_data_path, filename), aumented_img)
        
def split_data(processed_data_path, train_data_path, test_data_path):
    """
    将数据集划分为训练集和测试集
    Args:
        processed_data_path (_type_): _description_
        train_data_path (_type_): _description_
        test_data_path (_type_): _description_
    """
    train_data, test_data = train_test_split(os.listdir(processed_data_path), test_size=0.2, random_state=42)
    for filename in train_data:
        img = cv2.imread(os.path.join(processed_data_path, filename))
        cv2.imwrite(os.path.join(train_data_path, filename), img)
    for filename in test_data:
        img = cv2.imread(os.path.join(processed_data_path, filename))
        cv2.imwrite(os.path.join(test_data_path, filename), img)
        
if __name__ == '__main__':
    capture_data(raw_data_path)
    preprocess_data(raw_data_path, processed_data_path)
    split_data(processed_data_path, train_data_path, test_data_path)