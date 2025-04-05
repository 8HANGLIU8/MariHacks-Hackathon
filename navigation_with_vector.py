import cv2
from ultralytics import YOLO
import numpy as np
from pygame import mixer
import time

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Nano model; adjust as needed

# Initialize pygame mixer for sound
mixer.init()
sound = mixer.Sound("Stop.mp3")  # Replace with your sound file path (e.g., "alert.mp3")
soundright = mixer.Sound("right.mp3")  # Replace with your sound file path (e.g., "alert.mp3")
soundleft = mixer.Sound("left.mp3")  # Replace with your sound file path (e.g., "alert.mp3")
last_play_time = 0  # To prevent sound overlap
last_play_time2 = 0  # To prevent sound overlap
last_play_time3 = 0  # To prevent sound overlap
# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Distance estimation parameters (calibrate these for your setup)
FOCAL_LENGTH = 1500  # Approximate focal length in pixels (depends on your camera)
KNOWN_WIDTH = 0.3  # Known width of a reference object in meters (e.g., person's head ~30cm)
WARNING_DISTANCE = 0.4  # Warning threshold set to 40 cm (0.4 meters)

# Function to estimate distance based on bounding box width
def estimate_distance(box_width):
    # Distance = (Known Width * Focal Length) / Perceived Width
    return (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get frame dimensions and calculate center coordinates
    height, width = frame.shape[:2]
    center_x = width // 2  # Integer division for x-coordinate
    center_y = height // 2  # Integer division for y-coordinate

    # Perform object detection
    results = model(frame)

    # Flag for warning
    warning_triggered = False

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if confidence > 0.4:
                # Calculate bounding box width in pixels
                box_width = x2 - x1

                # Estimate distance
                distance = estimate_distance(box_width)

                # Draw bounding box and label
                color = (0, 255, 0)  # Green by default
                if distance < WARNING_DISTANCE:
                    color = (0, 0, 255)  # Red if within 40 cm
                    warning_triggered = True
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add cup detection and direction logic
                if label == "cup" or label == "mug" or label == "glass" or label == "bottle":
                    # Draw a circle at the center of the cup

                    cup_center_x = (x1 + x2) // 2  # x-coordinate of cup's center
                    delta_x = cup_center_x - center_x  # Change in x
                    threshold = width // 5  # 1/5 of screen width
                    #draw a line between the center of the screen and the cup
                    cv2.line(frame, (center_x, center_y), (cup_center_x, center_y), (255, 0, 0), 2)
                    if delta_x > threshold:
                        cv2.putText(frame, "Go to the right", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        current_time2 = time.time()
                        if current_time2 - last_play_time2 >= 2:  # 200 ms cooldown
                            soundright.play(maxtime=2000)  # Play for 200 ms
                            last_play_time2 = current_time2
                    elif delta_x < -threshold:
                        cv2.putText(frame, "Go to the left", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        current_time3 = time.time()
                        if current_time3 - last_play_time3 >= 2:  # 200 ms cooldown
                            soundleft.play(maxtime=2000)  # Play for 200 ms
                            last_play_time3 = current_time3

    # Draw a point at the center of the screen
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue dot, radius 5, filled

    # Display warning and play sound if an object is within 40 cm
    if warning_triggered:
        cv2.putText(frame, "WARNING: Too Close to Object!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # Play sound if it hasn’t been played in the last 0.2 seconds
        current_time = time.time()
        if current_time - last_play_time >= 2:  # 200 ms cooldown
            sound.play(maxtime=2000)  # Play for 200 ms
            last_play_time = current_time

    # Show the frame
    cv2.imshow("Object Detection with Proximity Warning", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
mixer.quit()
cap.release()
cv2.destroyAllWindows()
