import cv2
from ultralytics import YOLO


model = YOLO("/Users/snezhanakoleva/PycharmProject/VNP/runs/detect/my_project_v2_tuned/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream or file. Please check camera index and permissions.")
    exit()

cv2.namedWindow("YOLOv8 Real-time Object Detection", cv2.WINDOW_NORMAL)

print("Press 'q' to quit the detection window.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame, exiting...")
        break

    results = model(frame, conf=0.8, iou=0.8, verbose=False)

    if results:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    cv2.imshow("YOLOv8 Real-time Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")



#
# yolo detect train \
#   model=yolov8s.pt \
#   data=/Users/snezhanakoleva/PycharmProject/VNP/dataset/data.yaml \
#   epochs=250 \
#   imgsz=640 \
#   batch=16 \
#   name=my_project_v2_tuned \
#   patience=20 \
#   optimizer=AdamW \
#   lr0=0.001 \
#   lrf=0.01 \
#   degrees=20 \
#   scale=0.85 \
#   translate=0.2 \
#   shear=10 \
#   perspective=0.0005 \
#   hsv_h=0.015 \
#   hsv_s=0.3 \
#   hsv_v=0.2 \
#   mosaic=1.0 \
#   mixup=0.1 \
#   copy_paste=0.1 \
#   flipud=0.0 \
#   fliplr=0.5



# yolo detect predict \
#   model=runs/detect/my_project_small_dataset_tuned/weights/best.pt \
#   source=0 \
#   agnostic_nms=True \ conf = 0.8 \ iou = 0.8

#https://app.roboflow.com/vnp-rgviz/my-first-project-jbvmy/browse?queryText=split%3Avalid&pageSize=50&startingIndex=0&browseQuery=true