import cv2
import numpy as np
from ultralytics import YOLO

# Room and desk dimensions
room_width = 2.4  # meters
room_length = 9.3  # meters
desk_width = 0.5  # meters
desk_length = 0.6  # meters
cell_size = 0.1  # meters per cell
grid_width = int(room_width / cell_size)
grid_length = int(room_length / cell_size)

# Grid initialization
grid = np.zeros((grid_length, grid_width), dtype=np.uint8)

# Camera settings
focal_length = 1000
image_width = 1280
image_height = 720

# Desk memory
mapped_desks = []

# Orientation and motion tracking
yaw = 0.0  # facing north
camera_x = room_width / 2
camera_z = room_length

prev_frame_gray = None
prev_pts = None

# Load YOLO
model = YOLO("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video feed.")
        break

    image_height, image_width = frame.shape[:2]
    curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow motion tracking
    if prev_frame_gray is not None and prev_pts is not None:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_pts, None)
        good_curr = curr_pts[status == 1]
        good_prev = prev_pts[status == 1]

        if len(good_curr) > 0:
            dx = good_curr[:, 0] - good_prev[:, 0]
            dy = good_curr[:, 1] - good_prev[:, 1]
            yaw_change = np.mean(dx) * (np.pi / 180) * 0.1
            yaw += yaw_change
            print(f"Current yaw (degrees): {yaw * 180 / np.pi:.2f}")

            # Estimate movement
            dz = -np.mean(dy) * 0.01
            dx_move = -np.mean(dx) * 0.005
            dx_rot = dx_move * np.cos(yaw) - dz * np.sin(yaw)
            dz_rot = dx_move * np.sin(yaw) + dz * np.cos(yaw)

            camera_x += dx_rot
            camera_z += dz_rot
            camera_x = max(0, min(room_width, camera_x))
            camera_z = max(0, min(room_length, camera_z))

    # Update optical flow state
    prev_frame_gray = curr_frame_gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(curr_frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes
    desks = [box for box in detections if model.names[int(box.cls)] == 'dining table']
    print(f"Number of desks detected: {len(desks)}")

    for box in desks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        d = (focal_length * desk_width) / w
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        X = (u - image_width / 2) * d / focal_length
        Z = d

        X_rot = X * np.cos(yaw) - Z * np.sin(yaw)
        Z_rot = X * np.sin(yaw) + Z * np.cos(yaw)

        global_x = camera_x + X_rot
        global_z = camera_z + Z_rot

        duplicate = False
        for mapped_x, mapped_z in mapped_desks:
            dist = np.sqrt((global_x - mapped_x)**2 + (global_z - mapped_z)**2)
            if dist < 0.5:
                duplicate = True
                break

        if not duplicate:
            grid_x = int(global_x / cell_size)
            grid_z = int(global_z / cell_size)

            print(f"Desk at global (x, z): ({global_x:.2f}, {global_z:.2f})")
            print(f"Mapping to grid (x, z): ({grid_x}, {grid_z})")

            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_length:
                grid[grid_z, grid_x] = 1
                mapped_desks.append((global_x, global_z))
                print(f"Grid updated at ({grid_z}, {grid_x})")
            else:
                print(f"Grid indices out of bounds: ({grid_x}, {grid_z})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Desk', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Grid visualization
    grid_display = np.repeat(grid[:, :, np.newaxis], 3, axis=2) * 255
    grid_display = cv2.resize(grid_display, (grid_width * 10, grid_length * 10), interpolation=cv2.INTER_NEAREST)

    # Mark desks
    for i in range(grid_length):
        for j in range(grid_width):
            if grid[i, j] == 1:
                x_pos = j * 10 + 5
                y_pos = i * 10 + 5
                cv2.putText(grid_display, 'D', (x_pos - 5, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 🔴 Draw moving camera point
    grid_cam_x = int(camera_x / cell_size)
    grid_cam_z = int(camera_z / cell_size)
    if 0 <= grid_cam_x < grid_width and 0 <= grid_cam_z < grid_length:
        x_pos = grid_cam_x * 10 + 5
        y_pos = grid_cam_z * 10 + 5
        cv2.circle(grid_display, (x_pos, y_pos), 5, (0, 0, 255), -1)

    # Show frames
    cv2.imshow("iPhone Camera Feed", frame)
    cv2.namedWindow("Desk Map")
    cv2.moveWindow("Desk Map", 100, 100)
    cv2.imshow("Desk Map", grid_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
