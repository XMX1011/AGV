# src/utils.py
import cv2
import numpy as np
import torch
import json

def preprocess_image(image):
    # 去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 边缘增强
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    image = cv2.addWeighted(image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 1, 0)
    
    return image

def subpixel_interpolation(image, x, y):
    x_int, y_int = int(x), int(y)
    dx, dy = x - x_int, y - y_int
    if dx<0 or dx>=1 or dy<0 or dy>=1:
        return x, y
    if x_int +1 >= image.shape[1] and y_int +1 >= image.shape[0]:
        return x, y
    
    q11 = image[y_int, x_int]
    q21 = image[y_int, x_int+1]
    q12 = image[y_int+1, x_int]
    q22 = image[y_int+1, x_int+1]
    
    interpolated_value = (q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy)
    return interpolated_value

def multi_frame_fusion(frames):
    fused_frame = np.mean(frames, axis=0).astype(np.uint8)
    return fused_frame

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()

    avg_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")