import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Lightweight model trained on COCO dataset

# Access the iPhone camera
cap = cv2.VideoCapture(0)  # Using index 0 since it works for Continuity Camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video feed.")
        break

    # Run object detection
    results = model(frame)
    detections = results[0].boxes  # Extract bounding boxes

    # Filter for desks (using 'dining table' class as a proxy for desks)
    desks = [box for box in detections if model.names[int(box.cls)] == 'dining table']

    # Draw bounding boxes around detected desks
    for box in desks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Desk', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the live feed with detections
    cv2.imshow("iPhone Camera Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()