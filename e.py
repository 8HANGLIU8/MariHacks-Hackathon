
import pyttsx3




import cv2
from ultralytics import YOLO
import numpy as np
from pygame import mixer
import time
import base64
from openai import OpenAI
import threading
from queue import Queue

# Assuming API-KEY is defined in kaepyi module; replace with actual import if needed
from kaepyi import API_KEY  # Note: Fixed typo from "API-KEY" to "API_KEY"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)
CHECK_INTERVAL = 10  # Seconds between ChatGPT analyses

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

# Queue for ChatGPT analysis results
analysis_queue = Queue()

# Function to convert OpenCV frame to base64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# Function to analyze frame with ChatGPT (runs in background)
def analyze_frame_async(frame, queue):
    base64_image = frame_to_base64(frame)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I am a blind person. You're responsible for my safety. Then, I'll send you an image every 5 seconds. You don't have to respond when there is nothing going on. However, if there's something, if there's information relevant to my safety, please send me a message."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        if result:  # Only queue non-empty responses
            queue.put(result)
    except Exception as e:
        queue.put(f"ChatGPT Error: {e}")

# Distance estimation parameters (calibrate these for your setup)
FOCAL_LENGTH = 1500  # Approximate focal length in pixels (depends on your camera)
KNOWN_WIDTH = 0.3  # Known width of a reference object in meters (e.g., person's head ~30cm)
WARNING_DISTANCE = 0.4  # Warning threshold set to 40 cm (0.4 meters)

# Function to estimate distance based on bounding box width
def estimate_distance(box_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

# Main loop
last_check_time = 0  # For ChatGPT interval
current_analysis = ""  # Store latest ChatGPT message

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # ChatGPT analysis every 10 seconds (async)
    current_time = time.time()
    if current_time - last_check_time >= CHECK_INTERVAL:
        frame_copy = frame.copy()  # Avoid modifying the original frame
        thread = threading.Thread(target=analyze_frame_async, args=(frame_copy, analysis_queue))
        thread.start()
        last_check_time = current_time

    # Check for new ChatGPT analysis
    if not analysis_queue.empty():
        current_analysis = analysis_queue.get()
        print(f"ChatGPT Safety Message: {current_analysis}")

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
                    cup_center_x = (x1 + x2) // 2  # x-coordinate of cup's center
                    delta_x = cup_center_x - center_x  # Change in x
                    threshold = width // 5  # 1/5 of screen width
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
        current_time = time.time()
        if current_time - last_play_time >= 2:  # 200 ms cooldown
            sound.play(maxtime=2000)  # Play for 200 ms
            last_play_time = current_time

    # Display ChatGPT analysis if available

    """if current_analysis:
        cv2.putText(frame, current_analysis[:50], (10, 150),  # Truncate for display
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)"""

    if current_analysis:
    # Show on screen
        cv2.putText(frame, current_analysis[:50], (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    

    # Speak it out loud
    speak(current_analysis)

    # Show the frame
    cv2.imshow("Object Detection with Proximity Warning", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
mixer.quit()
cap.release()
cv2.destroyAllWindows()