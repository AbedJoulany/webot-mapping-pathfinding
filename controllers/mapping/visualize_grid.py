import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_occupancy_grid(file_path, map_size, resolution):
    """
    Converts lidar point clouds from a CSV file to an occupancy grid.

    Args:
    - file_path (str): Path to the CSV file containing lidar point clouds.
    - map_size (float): Size of the map in meters.
    - resolution (float): Size of each grid cell in meters.

    Returns:
    - occupancy_grid (np.ndarray): The generated occupancy grid.
    """
    # Step 1: Read the point clouds from the CSV file
    point_clouds = pd.read_csv(file_path, header=None)
    point_clouds.columns = ['x', 'y']

    # Step 2: Convert the point clouds to an occupancy grid
    grid_size = int(map_size / resolution)

    # Initialize the occupancy grid
    occupancy_grid = np.zeros((grid_size, grid_size))

    # Convert lidar points to grid coordinates
    for index, row in point_clouds.iterrows():
        x, y = row['x'], row['y']
        grid_x = int((x + map_size / 2) / resolution)
        grid_y = int((y + map_size / 2) / resolution)
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            occupancy_grid[grid_y, grid_x] = 1  # Mark cell as occupied

    return occupancy_grid

def visualize_occupancy_grid(occupancy_grid):
    """
    Visualizes the occupancy grid using matplotlib.

    Args:
    - occupancy_grid (np.ndarray): The occupancy grid to visualize.
    """
    # Step 3: Visualize the occupancy grid
    plt.imshow(occupancy_grid, cmap='Greys', origin='lower')
    plt.title('Occupancy Grid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def print_occupancy_grid(occupancy_grid):
    """
    Prints the occupancy grid to the console.

    Args:
    - occupancy_grid (np.ndarray): The occupancy grid to print.
    """
    for row in occupancy_grid:
        print(' '.join('1' if cell else '0' for cell in row))

# Usage example
file_path = 'map.csv'
map_size = 2  # meters
resolution = 0.001  # meter per cell

#occupancy_grid = create_occupancy_grid(file_path, map_size, resolution)
#visualize_occupancy_grid(occupancy_grid)
