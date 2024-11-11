# src/data.py
import os
import cv2
import numpy as np
import yaml

def collect_data(output_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = os.path.join(output_dir, f"image_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    output_dir = config['data']['processed_dir']
    os.makedirs(output_dir, exist_ok=True)
    collect_data(output_dir)