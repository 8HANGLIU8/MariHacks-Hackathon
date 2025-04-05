import cv2
import torch
from ultralytics import YOLO
from PIL import Image

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

    # Process detections and depth
    cell_height, cell_width = frame.shape[0] // grid_size, frame.shape[1] // grid_size
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = yolo_model.names[int(box.cls.item())]
        avg_depth = depth[y1:y2, x1:x2].mean()

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        grid_x, grid_y = center_x // cell_width, center_y // cell_height
        if avg_depth > 1000:  # Adjust threshold
            grid[grid_y][grid_x] = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {avg_depth:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
