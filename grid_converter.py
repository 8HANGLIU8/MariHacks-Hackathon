# Constants
room_width = 5.38   # Actual room width
room_length = 9.3    # Actual room length
desk_length = 0.6    # Actual desk length
desk_width = 0.45    # Actual desk width
cell_size = 0.4    # Initial cell size (smaller = better precision)

# Calculate grid dimensions based on room size and cell size
grid_width = int(room_width / cell_size)    # Number of cells along width
grid_length = int(room_length / cell_size)  # Number of cells along length

# Initialize the grid with zeros (0 = walkable, 1 = obstacle)
grid = [[0 for _ in range(grid_width)] for _ in range(grid_length)]
print(f"Initial grid ({grid_length}x{grid_width}, cell size = {cell_size}m):")
for row in grid:
    print(row)

# Sample desk coordinates (in meters, relative to room origin)
desk_coords = [(1, 1), (4, 4)]  # Example desk center points

# Function to mark rectangular desks on the grid
def mark_desk(grid, x, y, desk_length, desk_width, cell_size):
    # Convert desk center coordinates to grid indices
    grid_x_start = int(x / cell_size - desk_length / (2 * cell_size))
    grid_y_start = int(y / cell_size - desk_width / (2 * cell_size))
    grid_x_end = int(x / cell_size + desk_length / (2 * cell_size)) + 1
    grid_y_end = int(y / cell_size + desk_width / (2 * cell_size)) + 1

    # Ensure indices are within grid bounds
    grid_x_start = max(0, min(grid_x_start, grid_width - 1))
    grid_y_start = max(0, min(grid_y_start, grid_length - 1))
    grid_x_end = max(0, min(grid_x_end, grid_width))
    grid_y_end = max(0, min(grid_y_end, grid_length))

    # Mark the desk area as obstacles (1)
    for i in range(grid_y_start, grid_y_end):
        for j in range(grid_x_start, grid_x_end):
            grid[i][j] = 1

# Mark desks on the grid
for x, y in desk_coords:
    mark_desk(grid, x, y, desk_length, desk_width, cell_size)

# Print the grid with desks marked
print(f"\nGrid with desks marked (1 = desk, 0 = walkable, cell size = {cell_size}m):")
for row in grid:
    print(row)

# Option to adjust cell size for better precision
new_cell_size = float(input("Enter a new cell size (in meters, e.g., 0.25 for higher precision): "))
if 0 < new_cell_size < max(room_width, room_length):
    cell_size = new_cell_size
    grid_width = int(room_width / cell_size)
    grid_length = int(room_length / cell_size)
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_length)]
    for x, y in desk_coords:
        mark_desk(grid, x, y, desk_length, desk_width, cell_size)
    print(f"\nUpdated grid with new cell size {cell_size}m ({grid_length}x{grid_width}):")
    for row in grid:
        print(row)
else:
    print("Invalid cell size. Keeping original grid.")