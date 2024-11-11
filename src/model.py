# src/model.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from config import config
from src.data import load_data
from src.utils import train_one_epoch, validate

def train_model(config):
    model = YOLO('yolov5s')
    model.nc = config['model']['num_classes']  # 设置类别数
    model.names = ['nuts_not_removed', 'nuts_removed', 'holes_aligned']

    train_loader, val_loader = load_data(config)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = config['training']['device']
    model.to(device)

    for epoch in range(config['training']['epochs']):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        validate(model, val_loader, criterion, device)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models/checkpoint_epoch_{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    final_model_path = 'models/final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train_model(config)