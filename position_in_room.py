import cv2
import numpy as np

# Camera parameters
focal_length = 1000  # pixels
image_width = 1280   # pixels (landscape width)
image_height = 720   # pixels (landscape height)
fov_horizontal = 60  # degrees (approximate iPhone camera FOV in landscape)

# Cell size for the grid
cell_size = 0.4  # meters per cell

# Scaling factor for visualization
scale_factor = 20  # Pixels per meter

# Maximum grid dimensions to prevent memory issues
MAX_GRID_DIMENSION = 1000  # Maximum width/height in cells
MIN_GRID_DIMENSION = 10    # Minimum width/height in cells (default if no walls)

# Access the iPhone camera
cap = cv2.VideoCapture(0)  # Index 0 for Continuity Camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

# Variables for optical flow to estimate movement and rotation
prev_frame_gray = None
prev_pts = None
yaw = 0.0  # Camera orientation (radians)
initial_yaw = None  # To track when a full 360° scan is complete
scan_complete = False  # Flag to indicate when scanning is done

# Variables to store room geometry and camera position
wall_points = []  # List of (x1, z1, x2, z2) for walls in global coordinates
wall_labels = []  # Labels for walls ("Wall 1", "Wall 2", etc.)
camera_pos = (0.0, 0.0)  # (x, z) in meters, initially at origin
min_x, max_x = float('inf'), float('-inf')
min_z, max_z = float('inf'), float('-inf')

# Variables for debouncing wall detection
last_wall_frame = -1
cooldown_frames = 30  # Number of frames to wait before detecting a new wall

# Main loop: Detect walls as you scan
frame_count = 0
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video feed. Exiting...")
            break

        frame_count += 1

        # Check frame orientation and rotate if necessary (for landscape mode)
        frame_height, frame_width = frame.shape[:2]
        if frame_width < frame_height:  # Portrait orientation detected
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_height, frame_width = frame.shape[:2]

        # Ensure frame matches expected dimensions
        if frame_width != image_width or frame_height != image_height:
            frame = cv2.resize(frame, (image_width, image_height))

        # Convert to grayscale for optical flow and edge detection
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Estimate camera movement and rotation using optical flow
        if prev_frame_gray is not None and prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame_gray, curr_frame_gray, prev_pts, None
            )
            good_curr = curr_pts[status == 1]
            good_prev = prev_pts[status == 1]

            if len(good_curr) > 0:
                dx = np.mean(good_curr[:, 0] - good_prev[:, 0])
                dy = np.mean(good_curr[:, 1] - good_prev[:, 1])

                # Estimate yaw change (horizontal axis in landscape mode)
                yaw_change = dx * (np.pi / 180) * 0.1
                yaw += yaw_change

                # Normalize yaw to [0, 2π]
                yaw = yaw % (2 * np.pi)

                # Set initial yaw on first frame
                if initial_yaw is None:
                    initial_yaw = yaw

                # Check if a full 360° scan is complete
                if abs(yaw - initial_yaw) >= 2 * np.pi and not scan_complete:
                    scan_complete = True
                    # Connect the last wall to the first to close the polygon
                    if wall_points:
                        last_x2, last_z2 = wall_points[-1][2], wall_points[-1][3]
                        first_x1, first_z1 = wall_points[0][0], wall_points[0][1]
                        wall_points.append((last_x2, last_z2, first_x1, first_z1))
                        wall_labels.append(f"Wall {len(wall_points)}")

                # Estimate translation
                scale = 0.01
                dx_m = dx * scale
                dy_m = dy * scale
                camera_x = camera_pos[0] + dx_m * np.cos(yaw) + dy_m * np.sin(yaw)
                camera_z = camera_pos[1] - dx_m * np.sin(yaw) + dy_m * np.cos(yaw)
                camera_pos = (camera_x, camera_z)

        prev_frame_gray = curr_frame_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(
            curr_frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7
        )

        # Detect walls if scanning is not complete
        if not scan_complete and frame_count - last_wall_frame > cooldown_frames:
            # Downscale frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(small_gray, 50, 150, apertureSize=3)

            # Detect lines using Hough Transform (focus on floor-wall boundaries)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

            # Filter for potential wall-floor boundary (adjusted for landscape mode)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Scale back to original coordinates
                    x1, x2 = x1 * 2, x2 * 2
                    y1, y2 = y1 * 2, y2 * 2
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 0 <= angle <= 10 or 170 <= angle <= 180:
                        if y1 > image_height * 0.7 and y2 > image_height * 0.7:  # Adjusted for landscape mode
                            # Estimate distance using the y-coordinate of the line
                            y_base = (y1 + y2) / 2
                            d = (focal_length * 1.0) / (image_height - y_base)  # Assume camera height 1m

                            # Convert x-coordinates to camera coordinates
                            X1 = (x1 - image_width / 2) * d / focal_length
                            X2 = (x2 - image_width / 2) * d / focal_length
                            Z1 = Z2 = d

                            # Limit wall length to prevent radiating lines
                            wall_length = np.sqrt((X2 - X1)**2 + (Z2 - Z1)**2)
                            if wall_length > 5:  # Cap at 5 meters
                                scale_factor_wall = 5 / wall_length
                                X2 = X1 + (X2 - X1) * scale_factor_wall
                                Z2 = Z1 + (Z2 - Z1) * scale_factor_wall

                            # Rotate coordinates based on current yaw
                            X1_rot = X1 * np.cos(yaw) - Z1 * np.sin(yaw)
                            Z1_rot = X1 * np.sin(yaw) + Z1 * np.cos(yaw)
                            X2_rot = X2 * np.cos(yaw) - Z2 * np.sin(yaw)
                            Z2_rot = X2 * np.sin(yaw) + Z2 * np.cos(yaw)

                            # Transform to global coordinates
                            global_x1 = camera_pos[0] + X1_rot
                            global_z1 = camera_pos[1] + Z1_rot
                            global_x2 = camera_pos[0] + X2_rot
                            global_z2 = camera_pos[1] + Z2_rot

                            # Check for duplicates based on angle and distance
                            duplicate = False
                            if wall_points:
                                new_angle = np.arctan2(global_z2 - global_z1, global_x2 - global_x1)
                                last_x1, last_z1, last_x2, last_z2 = wall_points[-1]
                                last_angle = np.arctan2(last_z2 - last_z1, last_x2 - last_x1)
                                angle_diff = abs((new_angle - last_angle) % (2 * np.pi))
                                if angle_diff < np.pi / 6:  # Within 30 degrees
                                    dist1 = np.sqrt((global_x1 - last_x1)**2 + (global_z1 - last_z1)**2)
                                    if dist1 < 1.0:  # Threshold in meters
                                        duplicate = True

                            if not duplicate:
                                # Connect to the previous wall if exists
                                if wall_points:
                                    prev_x2, prev_z2 = wall_points[-1][2], wall_points[-1][3]
                                    global_x1, global_z1 = prev_x2, prev_z2

                                wall_points.append((global_x1, global_z1, global_x2, global_z2))
                                wall_labels.append(f"Wall {len(wall_points)}")
                                last_wall_frame = frame_count
                                break

        # Update room bounds
        min_x, max_x = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        for x1, z1, x2, z2 in wall_points:
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_z = min(min_z, z1, z2)
            max_z = max(max_z, z1, z2)

        # Ensure bounds are valid and have a minimum size
        if min_x == float('inf') or max_x == float('-inf'):
            min_x, max_x = -5.0, 5.0
        if min_z == float('inf') or max_z == float('-inf'):
            min_z, max_z = -5.0, 5.0
        # Enforce minimum room size to prevent zero grid dimensions
        if max_x - min_x < cell_size * MIN_GRID_DIMENSION:
            center_x = (min_x + max_x) / 2
            min_x = center_x - (cell_size * MIN_GRID_DIMENSION) / 2
            max_x = center_x + (cell_size * MIN_GRID_DIMENSION) / 2
        if max_z - min_z < cell_size * MIN_GRID_DIMENSION:
            center_z = (min_z + max_z) / 2
            min_z = center_z - (cell_size * MIN_GRID_DIMENSION) / 2
            max_z = center_z + (cell_size * MIN_GRID_DIMENSION) / 2

        # Limit bounds to prevent excessive grid size
        min_x = max(min_x, -50.0)
        max_x = min(max_x, 50.0)
        min_z = max(min_z, -50.0)
        max_z = min(max_z, 50.0)

        # Create grid display with dynamic size
        room_width = max_x - min_x
        room_length = max_z - min_z
        grid_width = int(room_width / cell_size)
        grid_length = int(room_length / cell_size)

        # Ensure grid dimensions are at least 1
        grid_width = max(MIN_GRID_DIMENSION, min(grid_width, MAX_GRID_DIMENSION))
        grid_length = max(MIN_GRID_DIMENSION, min(grid_length, MAX_GRID_DIMENSION))

        # Debug: Log grid dimensions
        print(f"Grid dimensions: {grid_width}x{grid_length}")

        grid_display = np.zeros((grid_length * scale_factor, grid_width * scale_factor, 3), dtype=np.uint8)

        # Draw walls as white lines
        for i, (x1, z1, x2, z2) in enumerate(wall_points):
            pt1_x = int((x1 - min_x) / cell_size * scale_factor)
            pt1_z = int((z1 - min_z) / cell_size * scale_factor)
            pt2_x = int((x2 - min_x) / cell_size * scale_factor)
            pt2_z = int((z2 - min_z) / cell_size * scale_factor)

            # Clamp coordinates to grid dimensions
            pt1_x = max(0, min(pt1_x, grid_width * scale_factor - 1))
            pt1_z = max(0, min(pt1_z, grid_length * scale_factor - 1))
            pt2_x = max(0, min(pt2_x, grid_width * scale_factor - 1))
            pt2_z = max(0, min(pt2_z, grid_length * scale_factor - 1))

            cv2.line(grid_display, (pt1_x, pt1_z), (pt2_x, pt2_z), (255, 255, 255), 5)

            # Draw wall label
            label_pos_x = (pt1_x + pt2_x) // 2
            label_pos_z = (pt1_z + pt2_z) // 2
            label_pos_x = max(0, min(label_pos_x, grid_width * scale_factor - 1))
            label_pos_z = max(0, min(label_pos_z, grid_length * scale_factor - 1))
            cv2.putText(
                grid_display, wall_labels[i], (label_pos_x, label_pos_z - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )

        # Draw camera position as a red dot
        cam_x_grid = int((camera_pos[0] - min_x) / cell_size * scale_factor)
        cam_z_grid = int((camera_pos[1] - min_z) / cell_size * scale_factor)
        cam_x_grid = max(0, min(cam_x_grid, grid_width * scale_factor - 1))
        cam_z_grid = max(0, min(cam_z_grid, grid_length * scale_factor - 1))
        cv2.circle(grid_display, (cam_x_grid, cam_z_grid), 5, (0, 0, 255), -1)

        # Display the camera feed and grid
        cv2.imshow("iPhone Camera Feed", frame)
        cv2.namedWindow("Room Map")
        cv2.moveWindow("Room Map", 100, 100)
        cv2.imshow("Room Map", grid_display)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()