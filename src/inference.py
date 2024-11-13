# src/inference.py
import cv2
import numpy as np
import torch
import yaml
import json
import torchvision.transforms as transforms
from ultralytics import YOLO
from config import config
from src.utils import preprocess_image, subpixel_interpolation, multi_frame_fusion
from src.model import BoltHoleModel


def load_model(model_path, config):
    model = BoltHoleModel(num_classes=config['num_classes'].to(config['device']))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def find_nearest_pairs(bolts, holes):
    pairs = []
    for bolt in bolts:
        min_dist = float('inf')
        nearest_hole = None
        for hole in holes:
            dist = np.linalg.norm(np.array(bolt) - np.array(hole))
            if dist < min_dist:
                min_dist = dist
                nearest_hole = hole
        pairs.append((bolt, nearest_hole, min_dist))
    return pairs

def draw_annotations(image, bolts, holes, pairs, show_pairs=False):
    for bolt in bolts:
        cv2.point(image, tuple(map(int, bolt)), 5, (0, 255, 0), -1)  # 绿色表示螺栓
    for hole in holes:
        cv2.circle(image, tuple(map(int, hole)), 5, (0, 0, 255), -1)  # 红色表示安装孔
    if show_pairs:
        for bolt, hole, _ in pairs:
            cv2.dottedLine(image, tuple(map(int, bolt)), tuple(map(int, hole)), (255, 0, 0), 2)  # 蓝色虚线连接螺栓和安装孔

def check_alignment(pairs, threshold=10):
    aligned_pairs = []
    unaligned_pairs = []
    for bolt, hole, distance in pairs:
        if distance <= threshold:
            aligned_pairs.append((bolt, hole))
        else:
            unaligned_pairs.append((bolt, hole))
    return aligned_pairs, unaligned_pairs

def run_inference(model_path, config):
    model = load_model(model_path, config)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()

    # 未拆完、拆完未对齐、拆完已对齐
    class_names = ['not_removed', 'removed_not_aligned', 'removed_aligned']
    transform = transforms.Compose([
            transforms.resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frames = []
    bolts = []
    holes = []
    pairs = []
    show_pairs = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break 
        frames.append(frame)
        if len(frames) > 5:
            frames.pop(0)
        fused_frame = multi_frame_fusion(frames)
        preprocessed_frame = preprocess_image(fused_frame)
        input_tensor = transform(preprocessed_frame).unsqueeze(0).to(config['device'])
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
        
        


        predictions = []

        bolts = []
        holes = []
        
        if predicted_class == 1:
            x1, y1, x2, y2, conf, cls = predictions[0]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_subpixel = subpixel_interpolation(fused_frame, x_center, y_center)
            y_center_subpixel = subpixel_interpolation(fused_frame, y_center, x_center)
            bolts.append((x_center_subpixel, y_center_subpixel))
        elif predicted_class == 2:
            x1, y1, x2, y2, conf, cls = predictions[0]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_subpixel = subpixel_interpolation(fused_frame, x_center, y_center)
            y_center_subpixel = subpixel_interpolation(fused_frame, y_center, x_center)
            holes.append((x_center_subpixel, y_center_subpixel))
        
        pairs = find_nearest_pairs(bolts, holes)
        aligned_pairs, unaligned_pairs = check_alignment(pairs, threshold=10)

        if aligned_pairs:
            print("Aligned pairs:", aligned_pairs)
        if unaligned_pairs:
            print("Unaligned pairs:", unaligned_pairs)
        
        draw_annotations(fused_frame, bolts, holes, pairs, show_pairs=show_pairs)
        
        # 显示图像
        cv2.imshow('Frame', fused_frame)

        # 按 'p' 键切换显示配对
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            show_pairs = not show_pairs
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    run_inference('models/final_model.pt', config)
