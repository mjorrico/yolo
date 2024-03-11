import os
import ultralytics
from glob import glob
from ultralytics import YOLO
import sys
import numpy as np

if len(sys.argv) != 4:
    raise RuntimeError("Expected usage: python3 infer.py <source_dir> <destination_dir> <model_type>")
else:
    source_dir, destination_dir, modeltype= sys.argv[1:]
    source_files = os.path.join(source_dir, "*")
    print(f"Reading files from {source_dir}")
    print(f"Writing output to {destination_dir}")

videoformats = ["mp4", "mkv", "avi"]
imageformats = ["jpg", "png"]
files = glob(source_files)
images = [f for f in files if f[-3:] in imageformats]
videos = [f for f in files if f[-3:] in videoformats]

if modeltype == "truck":
    yolo = YOLO(model="runs/detect/yolov8_truck_best/weights/best.pt")
elif modeltype == "ambulance":
    yolo = YOLO(model="runs/detect/yolov8_ambulance_best/weights/best.pt")
else:
    raise ValueError(f"Model type {modeltype} is unknown. Expected model type: truck / ambulance.")

for img in images:
    yolo.predict(img, save=True, project=os.getcwd(), name=destination_dir, exist_ok=True)

for vid in videos:
    yolo.predict(vid, save=True, project=os.getcwd(), name=destination_dir, exist_ok=True)
