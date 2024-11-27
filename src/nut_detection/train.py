# 用于深度学习算法的训练，输出结果到models
# 使用config文件夹下面的det_config.yaml文件作为训练参数

import os
from torchvison import transforms
from torchvison import models
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
import tensorrt as trt

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        # 加载数据集
    train_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(config['train_data_path'], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    # 加载模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    for epoch in range(config['num_epochs']):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # 保存模型
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), os.path.join('models', 'nut_detection.pth'))
    print('Model saved!')
    # 导出onnx模型
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, os.path.join('models', 'nut_detection.onnx'), verbose=True)
    print('ONNX model exported!')
    # 导出tensorRT模型
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
    print('TensorRT model exported!')
    

if __name__ == '__main__':
    train('config/det_config.yaml')