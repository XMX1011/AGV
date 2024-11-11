# src/inference.py
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from config import config
from src.utils import preprocess_image, subpixel_interpolation, multi_frame_fusion

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
        pairs.append((bolt, nearest_hole))
    return pairs

def draw_annotations(image, bolts, holes, pairs, show_pairs=False):
    for bolt in bolts:
        cv2.circle(image, tuple(map(int, bolt)), 5, (0, 0, 255), -1)  # 红色圆圈表示螺栓
    for hole in holes:
        cv2.circle(image, tuple(map(int, hole)), 5, (0, 255, 0), -1)  # 绿色圆圈表示安装孔
    if show_pairs:
        for bolt, hole in pairs:
            cv2.line(image, tuple(map(int, bolt)), tuple(map(int, hole)), (255, 0, 0), 2)  # 蓝色线条连接螺栓和安装孔

def run_inference(model_path, config):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()

    class_names = ['nuts_not_removed', 'nuts_removed', 'holes_aligned']
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
        frame_resized = cv2.resize(preprocessed_frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_transposed = frame_normalized.transpose((2, 0, 1)).astype(np.float32)

        # 进行推理
        results = model(frame_resized)
        predictions = results[0].boxes.data.cpu().numpy()

        bolts = []
        holes = []

        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_subpixel = subpixel_interpolation(frame_resized, x_center, y_center)
            y_center_subpixel = subpixel_interpolation(frame_resized, y_center, x_center)

            # 绘制检测框和标签
            cv2.rectangle(fused_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            result_label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.putText(fused_frame, result_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_names[int(cls)] == 'nuts_removed':
                bolts.append((x_center_subpixel, y_center_subpixel))
            elif class_names[int(cls)] == 'holes_aligned':
                holes.append((x_center_subpixel, y_center_subpixel))

        pairs = find_nearest_pairs(bolts, holes)
        draw_annotations(fused_frame, bolts, holes, pairs, show_pairs)

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