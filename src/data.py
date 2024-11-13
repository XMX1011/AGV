# src/data.py
import os
import cv2
import numpy as np
import yaml
import json

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
    
def annotate_data(input_dir, output_dir):
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]
    for image_path in images:
        image = cv2.imread(image_path)
        label_path = os.path.join(output_dir,f'{os.path.basename(image_path)[:-4]}.json')
        
        annotation = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [
                    {"label":"not_removed","points":[[x1,y1],[x2,y2]],"group_id":None,"shape_type":"rectangle","flags":{}},
                    {"label":"removed_not_aligned","points":[[x3,y3],[x4,y4]],"group_id":None,"shape_type":"rectangle","flags":{}},
                    {"label":"removed_aligned","points":[[x5,y5],[x6,y6]],"group_id":None,"shape_type":"rectangle","flags":{}}
                ],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": image.shape[0],
            "imageWidth": image.shape[1]
        }
        
        with open(label_path, 'w') as f:
            json.dump(annotation, f,indent=4)
        
        print(f"Annotated and saved to {label_path}")

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    output_dir = config['data']['processed_dir']
    os.makedirs(output_dir, exist_ok=True)
    collect_data('data/raw')
    annotate_data('data/raw', 'data/labels')