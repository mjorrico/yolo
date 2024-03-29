# YOLO v8: Trucks and Ambulance Detection

## Inference

Run the following command to perform inference using photos and videos contained in your specified directory.

    $ python3 infer.py <source_dir> <destination_dir> <truck|ambulance>

Example: The following command will perform inference on all media files inside `input-sample` and save outputs in `my_images_processed` using `truck` YOLOv8 model.

    $ python3 infer.py input-sample my_images_processed truck

## Training

Model is trained using pretrained `yolov8m.pt` model from Ultralytics.

1. Prepare datasets. Follow this instruction [README.md](datasets/README.md).
2. Configure `path:` values in `truck-dataset.yaml` and `ambulance-dataset.yaml` to point them to `datasets/truck-dataset` and `datasets/ambulance-dataset` respectively.
3. Configure `truck.ipynb` and `ambulance.ipynb` accordingly.