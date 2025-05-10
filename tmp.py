import cv2
from ultralytics import YOLO

model = YOLO("/Users/snezhanakoleva/PycharmProject/VNP/runs/detect/my_project5/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model(frame)

    for result in results:
        result.plot()

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# yolo detect predict \
#   model=runs/detect/my_project/weights/best.pt \
#   source=0 \
#   agnostic_nms=True \ conf = 0.6 \ iou = 0.8

#https://app.roboflow.com/vnp-rgviz/my-first-project-jbvmy/browse?queryText=split%3Avalid&pageSize=50&startingIndex=0&browseQuery=true