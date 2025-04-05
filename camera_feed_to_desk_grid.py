import cv2
import numpy as np
from ultralytics import YOLO

# Provided constants
room_width = 2.4  # meters
room_length = 9.3  # meters
desk_width = 0.5  # meters
desk_length = 0.6  # meters
cell_size = 0.1    # meters per cell (provided value)
grid_width = int(room_width / cell_size)   # Number of cells along width
grid_length = int(room_length / cell_size) # Number of cells along length

# Initialize the global 2D grid (rows = z/depth, cols = x/width)
grid = np.zeros((grid_length, grid_width), dtype=np.uint8)

# Camera parameters (approximate, need calibration for accuracy)
focal_length = 1000  # pixels (adjust if needed)
image_width = 1280   # pixels (adjust to your camera resolution)
image_height = 720   # pixels (adjust to your camera resolution)

# List to store mapped desk positions (to avoid duplicates)
mapped_desks = []

# Camera orientation (yaw angle in radians)
yaw = 0.0  # Start facing "north" (0 degrees)

# Variables for optical flow to estimate rotation
prev_frame_gray = None
prev_pts = None

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Lightweight model trained on COCO dataset

# Access the iPhone camera
cap = cv2.VideoCapture(0)  # Index 0 for Continuity Camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video feed.")
        break

    # Update image dimensions based on actual frame size
    image_height, image_width = frame.shape[:2]

    # Convert frame to grayscale for optical flow
    curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Estimate camera rotation using optical flow
    if prev_frame_gray is not None and prev_pts is not None:
        # Compute optical flow to track feature points
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_pts, None
        )
        # Filter for successfully tracked points
        good_curr = curr_pts[status == 1]
        good_prev = prev_pts[status == 1]

        if len(good_curr) > 0:
            # Calculate angular displacement to estimate yaw change
            dx = good_curr[:, 0] - good_prev[:, 0]
            dy = good_curr[:, 1] - good_prev[:, 1]
            # Approximate yaw change (in radians) based on horizontal displacement
            yaw_change = np.mean(dx) * (np.pi / 180) * 0.1  # Scale factor (adjust as needed)
            yaw += yaw_change
            print(f"Current yaw (degrees): {yaw * 180 / np.pi:.2f}")

    # Update previous frame and detect new feature points
    prev_frame_gray = curr_frame_gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(
        curr_frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7
    )

    # Run object detection with YOLOv8
    results = model(frame)
    detections = results[0].boxes  # Extract bounding boxes

    # Filter for desks (using 'dining table' as a proxy)
    desks = [box for box in detections if model.names[int(box.cls)] == 'dining table']
    print(f"Number of desks detected: {len(desks)}")

    # Process each detected desk
    for box in desks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1  # Width in pixels

        # Estimate distance from camera using bounding box size
        d = (focal_length * desk_width) / w  # Depth (Z) in meters

        # Compute center of bounding box in pixel coordinates
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        # Convert to 3D camera coordinates (X, Y, Z)
        X = (u - image_width / 2) * d / focal_length  # Horizontal position
        Z = d                                          # Depth

        # Rotate coordinates based on current yaw
        X_rot = X * np.cos(yaw) - Z * np.sin(yaw)
        Z_rot = X * np.sin(yaw) + Z * np.cos(yaw)

        # Transform to global coordinates (camera at center of room)
        global_x = (room_width / 2) + X_rot
        global_z = (room_length / 2) + Z_rot

        # Check for duplicates
        duplicate = False
        for mapped_x, mapped_z in mapped_desks:
            dist = np.sqrt((global_x - mapped_x)**2 + (global_z - mapped_z)**2)
            if dist < 0.5:  # Threshold in meters (adjust as needed)
                duplicate = True
                break

        if not duplicate:
            # Map to grid indices
            grid_x = int(global_x / cell_size)
            grid_z = int(global_z / cell_size)

            print(f"Desk at global (x, z): ({global_x:.2f}, {global_z:.2f})")
            print(f"Mapping to grid (x, z): ({grid_x}, {grid_z})")

            # Mark the grid cell if within bounds
            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_length:
                grid[grid_z, grid_x] = 1  # Mark as occupied
                mapped_desks.append((global_x, global_z))
                print(f"Grid updated at ({grid_z}, {grid_x})")
            else:
                print(f"Grid indices out of bounds: ({grid_x}, {grid_z})")

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, 'Desk', (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Visualize the grid (scale up for visibility)
    grid_display = np.repeat(grid[:, :, np.newaxis], 3, axis=2) * 255  # Convert to RGB
    grid_display = cv2.resize(
        grid_display, (grid_width * 10, grid_length * 10), 
        interpolation=cv2.INTER_NEAREST
    )

    # Add grid labels for better visualization
    for i in range(grid_length):
        for j in range(grid_width):
            if grid[i, j] == 1:
                x_pos = j * 10 + 5  # Center of cell
                y_pos = i * 10 + 5
                cv2.putText(
                    grid_display, 'D', (x_pos - 5, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )

    # Display the camera feed and grid
    cv2.imshow("iPhone Camera Feed", frame)
    cv2.namedWindow("Desk Map")
    cv2.moveWindow("Desk Map", 100, 100)  # Ensure window is on-screen
    cv2.imshow("Desk Map", grid_display)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()