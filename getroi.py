import cv2
from ultralytics import YOLO
import numpy as np
from playsound import playsound
import threading

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Nano model; swap for other versions if needed

sound_playing = False
sound_lock = threading.Lock()
# Function to play sound in a separate thread
def play_sound():
    global sound_playing
    while sound_playing:
        playsound("1stop-it-audio-clip-100732.mp3")  # Replace "alert.mp3" with the path to your sound file
# Function to let the user select an ROI
def select_roi(frame):
    print("Select the Region of Interest by dragging a rectangle. Press 'Enter' to confirm.")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi  # Returns (x, y, w, h)

# Function to check if a bounding box is inside the ROI
def is_inside_roi(box, roi):
    box_x1, box_y1, box_x2, box_y2 = box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]
    return (box_x1 >= roi_x1 and box_x2 <= roi_x2 and 
            box_y1 >= roi_y1 and box_y2 <= roi_y2)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
if not cap.isOpened():
    raise ValueError("Error: Could not open webcam.")

# Read the first frame to select ROI
ret, frame = cap.read()
if not ret:
    raise ValueError("Error: Could not read from webcam.")
roi = select_roi(frame.copy())

# Main loop for live video processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference on the current frame
    results = model(frame)

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            confidence = confidences[i]

            # COCO dataset: "dining table" (class ID 60) as proxy for desks/tables
            if class_id == 60:  # Adjust if using a custom model with different class IDs
                # Default color for bounding boxes (blue)
                color = (255, 0, 0)  # BGR format

                # Change color to green if the box is inside the ROI
                if is_inside_roi((x1, y1, x2, y2), roi):
                    color = (0, 255, 0)  # Green for boxes in ROI
                    # Start playing sound if not already playing
                    with sound_lock:
                        if not sound_playing:
                            sound_playing = True
                            threading.Thread(target=play_sound, daemon=True).start()
                else:
                    # Stop playing sound if the box is outside the ROI
                    with sound_lock:
                        sound_playing = False

                # Draw the bounding box around the detected table/desk
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Table/Desk: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the ROI rectangle on the frame (optional, for visualization)
    roi_x, roi_y, roi_w, roi_h = roi
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 1)  # Red rectangle

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection with ROI", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()