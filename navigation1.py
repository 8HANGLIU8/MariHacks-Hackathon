import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from heapq import heappush, heappop
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open camera")

# Load models
yolo_model = YOLO('yolov8n.pt')
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas.to(device)

# MiDaS transform
midas_transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grid setup
grid_size = 10
grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

# A* pathfinding
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    if not (0 <= start[0] < grid_size and 0 <= start[1] < grid_size) or \
       not (0 <= goal[0] < grid_size and 0 <= goal[1] < grid_size) or \
       grid[goal[0]][goal[1]] == 1:
        return []  # Invalid start/goal or goal is obstacle

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse path

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                if grid[neighbor[0]][neighbor[1]] == 1:  # Obstacle
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
    return []  # No path found

# Define start and goal positions
start_pos = (grid_size - 1, grid_size // 2)  # Bottom center (user position)
goal_pos = (0, grid_size // 2)  # Top center (target further away)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reset grid each frame
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    # YOLOv8 detection
    yolo_results = yolo_model(frame)
    detections = yolo_results[0].boxes

    # MiDaS depth
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = midas_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = midas(img_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

    # Process detections and depth (only desks)
    cell_height, cell_width = frame.shape[0] // grid_size, frame.shape[1] // grid_size
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = yolo_model.names[int(box.cls.item())]
        if class_name == "dining table":  # Filter for desks/tables
            avg_depth = depth[y1:y2, x1:x2].mean()
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            grid_x, grid_y = center_x // cell_width, center_y // cell_height

            # Mark as obstacle if close enough (adjust threshold)
            DEPTH_THRESHOLD = 1000  # Adjust based on MiDaS output
            if avg_depth < DEPTH_THRESHOLD:  # Closer objects are obstacles
                if 0 <= grid_y < grid_size and 0 <= grid_x < grid_size:
                    grid[grid_y][grid_x] = 1

            # Draw box (red if close, green if far)
            color = (0, 0, 255) if avg_depth < DEPTH_THRESHOLD else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Table: {avg_depth:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Find path
    path = astar(grid, start_pos, goal_pos)

    # Draw path on frame
    for i in range(len(path) - 1):
        curr_y, curr_x = path[i]
        next_y, next_x = path[i + 1]
        pixel_curr = (curr_x * cell_width + cell_width // 2, curr_y * cell_height + cell_height // 2)
        pixel_next = (next_x * cell_width + cell_width // 2, next_y * cell_height + cell_height // 2)
        cv2.line(frame, pixel_curr, pixel_next, (255, 0, 0), 2)  # Blue path

    # Mark start and goal
    start_pixel = (start_pos[1] * cell_width + cell_width // 2, start_pos[0] * cell_height + cell_height // 2)
    goal_pixel = (goal_pos[1] * cell_width + cell_width // 2, goal_pos[0] * cell_height + cell_height // 2)
    cv2.circle(frame, start_pixel, 5, (0, 255, 255), -1)  # Yellow start
    cv2.circle(frame, goal_pixel, 5, (255, 255, 0), -1)  # Cyan goal

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()