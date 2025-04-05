import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

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
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
midas_transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grid setup
grid_size = 10
grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection (optional, for visualization)
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

    # Estimate wall depth as the 95th percentile (or max) of depth values
    wall_depth = np.percentile(depth, 95)  # 95th percentile for robustness
    # Alternatively, use np.max(depth) for the absolute farthest point

    # Process grid based on wall depth
    cell_height, cell_width = frame.shape[0] // grid_size, frame.shape[1] // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            # Define grid cell boundaries
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            # Get depth for this cell
            cell_depth = depth[y_start:y_end, x_start:x_end]
            avg_cell_depth = cell_depth.mean()
            # Update grid if cell depth is close to wall depth (within a tolerance)
            depth_tolerance = 0.1 * wall_depth  # 10% tolerance
            if abs(avg_cell_depth - wall_depth) < depth_tolerance:
                grid[i][j] = 1
            else:
                grid[i][j] = 0

    # Draw YOLO detections (optional, for reference)
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = yolo_model.names[int(box.cls.item())]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display wall depth on frame
    cv2.putText(frame, f"Wall Depth: {wall_depth:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()