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
focal_length = 1000  # Placeholder; calibrate your camera for accuracy
image_width = 1280
image_height = 720
min_feature_quality = 0.1
proximity_threshold = 0.5  # meters; warn if closer than this to a desk

# Desk memory and confidence threshold
mapped_desks = []
confidence_threshold = 0.7

# Orientation and motion tracking
yaw = 0.0  # Facing north (radians)
camera_x = room_width / 2
camera_z = room_length
motion_smoothing = 0.9

prev_frame_gray = None
prev_pts = None

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

# FPS counter
prev_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video feed.")
        break

    curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow motion tracking
    if prev_frame_gray is not None and prev_pts is not None:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        good_curr = curr_pts[status == 1]
        good_prev = prev_pts[status == 1]

        if len(good_curr) > 5:
            dx = np.median(good_curr[:, 0] - good_prev[:, 0])
            dy = np.median(good_curr[:, 1] - good_prev[:, 1])
            yaw_change = dx * (np.pi / 180) * 0.05
            yaw = motion_smoothing * yaw + (1 - motion_smoothing) * (yaw + yaw_change)

            dz = -dy * 0.005
            dx_move = -dx * 0.0025
            dx_rot = dx_move * np.cos(yaw) - dz * np.sin(yaw)
            dz_rot = dx_move * np.sin(yaw) + dz * np.cos(yaw)

            camera_x = motion_smoothing * camera_x + (1 - motion_smoothing) * (camera_x + dx_rot)
            camera_z = motion_smoothing * camera_z + (1 - motion_smoothing) * (camera_z + dz_rot)
            camera_x = max(0, min(room_width, camera_x))
            camera_z = max(0, min(room_length, camera_z))

            print(f"Camera position: ({camera_x:.2f}, {camera_z:.2f}), Yaw: {yaw * 180 / np.pi:.2f}°")

    # Update optical flow state
    prev_frame_gray = curr_frame_gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(
        curr_frame_gray, maxCorners=200, qualityLevel=min_feature_quality, minDistance=5
    )

    # Run YOLO detection
    results = model(frame, conf=confidence_threshold)
    detections = results[0].boxes
    desks = [box for box in detections if model.names[int(box.cls)] == 'dining table']

    for box in desks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf.item()
        w = x2 - x1
        d = (focal_length * desk_width) / w  # Depth estimation
        u = (x1 + x2) / 2
        X = (u - image_width / 2) * d / focal_length
        Z = d

        # Check proximity and warn if too close
        if d < proximity_threshold:
            print(f"WARNING: Too close to desk! Distance: {d:.2f}m")
            cv2.putText(frame, f"WARNING: Too close ({d:.2f}m)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Transform to global coordinates
        X_rot = X * np.cos(yaw) - Z * np.sin(yaw)
        Z_rot = X * np.sin(yaw) + Z * np.cos(yaw)
        global_x = camera_x + X_rot
        global_z = camera_z + Z_rot

        # Check for duplicates
        duplicate = False
        for mapped_x, mapped_z in mapped_desks:
            dist = np.sqrt((global_x - mapped_x)**2 + (global_z - mapped_z)**2)
            if dist < 0.5:
                duplicate = True
                break

        if not duplicate:
            grid_x = int(global_x / cell_size)
            grid_z = int(global_z / cell_size)

            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_length:
                grid[grid_z, grid_x] = 1
                mapped_desks.append((global_x, global_z))
                print(f"New desk at ({global_x:.2f}, {global_z:.2f}), Grid: ({grid_x}, {grid_z})")
            else:
                print(f"Desk out of bounds: ({grid_x}, {grid_z})")

        # Draw detection with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Desk ({confidence:.2f})'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Grid visualization
    grid_display = np.repeat(grid[:, :, np.newaxis], 3, axis=2) * 255
    grid_display = cv2.resize(grid_display, (grid_width * 20, grid_length * 20), interpolation=cv2.INTER_NEAREST)

    # Mark desks and camera
    for i in range(grid_length):
        for j in range(grid_width):
            if grid[i, j] == 1:
                x_pos = j * 20 + 10
                y_pos = i * 20 + 10
                cv2.circle(grid_display, (x_pos, y_pos), 5, (0, 255, 0), -1)

    grid_cam_x = int(camera_x / cell_size)
    grid_cam_z = int(camera_z / cell_size)
    if 0 <= grid_cam_x < grid_width and 0 <= grid_cam_z < grid_length:
        x_pos = grid_cam_x * 20 + 10
        y_pos = grid_cam_z * 20 + 10
        cv2.circle(grid_display, (x_pos, y_pos), 7, (0, 0, 255), -1)

    # FPS display
    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frames
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Desk Map", grid_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        grid.fill(0)
        mapped_desks.clear()
        camera_x, camera_z, yaw = room_width / 2, room_length, 0.0
        print("Reset grid and camera position.")

cap.release()
cv2.destroyAllWindows()

# Save grid map on exit
cv2.imwrite("desk_map.png", grid_display)
print("Desk map saved as 'desk_map.png'")