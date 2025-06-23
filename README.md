# Object Detection Model with YOLOv8

This repository contains code and resources for training a custom object detection model using YOLOv8. The trained model is intended to run on a Raspberry Pi-powered robot for real-time object detection.

## Model Setup
The model is based on YOLOv8n (nano version) for lightweight deployment on edge devices like Raspberry Pi.
### Framework
- Model: YOLOv8n (nano)
- Library: Ultralytics YOLO
- Target Device: Raspberry Pi for real-time inference

## Dataset
The project uses a custom dataset formatted for YOLO training with bounding boxes in normalized coordinates.
### Folder Structure
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
- images/ — contains training and validation images
- labels/ — contains corresponding YOLO-format annotation .txt files

## Model Visualization
To see the model performance visually, go to:
[Object-detection-model-metrics](https://github.com/kolevas/Object-detection-model/tree/main/runs/detect/model_nano)
