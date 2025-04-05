# Assume a 10x10 grid for a 10m x 10m room
grid_size = 10
grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
desk_coords = [(2.5, 3.1), (5.0, 1.2)]  # Example desk coordinates
for x, y in desk_coords:
    grid_x = int(x)  # Scale and round to grid cell
    grid_y = int(y)
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        grid[grid_y][grid_x] = 1  # Mark as obstacle
print(grid)