import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from scipy.ndimage import shift
import time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load YOLOv5 (nano for speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Load MiDaS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
transform = Compose([Resize((256, 256)), ToTensor()])

# Grid setup
grid_size = (20, 20)
grid = np.zeros(grid_size, dtype=int)
fov_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# User position
user_pos = (grid_size[0] - 1, grid_size[1] // 2)

# A* placeholder
def astar(grid, start, goal):
    return [start, goal]

# FPS tracking
prev_time = time.time()
frame_count = 0
fps = 0

# Depth threshold for "close" tables
DEPTH_THRESHOLD = 0.5  # Adjust this (0 to 1) based on your needs

# Video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    # YOLO inference
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    obstacles = []

    # MiDaS depth
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if torch.cuda.is_available() else torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize

    # Process detections, only for tables
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        label = model.names[int(cls)]
        if conf > 0.5 and label == "dining table" or label == "desk":  # YOLOv5 label for table
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)
            
            # Calculate depth for this table
            if y_max > y_min and x_max > x_min:
                obstacle_depth = np.mean(depth[y_min:y_max, x_min:x_max])
            else:
                obstacle_depth = 0
            
            obstacles.append({
                'label': label,
                'bbox': (x_min, y_min, x_max, y_max),
                'distance': obstacle_depth
            })

            # Determine box color based on depth
            color = (0, 255, 0) if obstacle_depth > DEPTH_THRESHOLD else (0, 0, 255)  # Green if far, red if close
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"Table: {obstacle_depth:.2f}"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update grid (still includes all obstacles for navigation)
    grid = shift(grid, [-1, 0], cval=0)
    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle['bbox']
        distance = obstacle['distance']
        grid_x = int((x_min + x_max) / 2 * grid_size[1] / fov_width)
        grid_y = int(distance * grid_size[0])
        grid_x = min(max(grid_x, 0), grid_size[1] - 1)
        grid_y = min(max(grid_y, 0), grid_size[0] - 1)
        grid[grid_y, grid_x] = 1

    # Update user position
    user_pos = (max(user_pos[0] - 1, 0), user_pos[1])

    # Compute path
    goal = (0, grid_size[1] // 2)
    path = astar(grid, user_pos, goal)

    # Calculate and display FPS
    frame_count += 1
    curr_time = time.time()
    if curr_time - prev_time >= 1.0:
        fps = frame_count / (curr_time - prev_time)
        frame_count = 0
        prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()