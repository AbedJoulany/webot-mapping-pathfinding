import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import binary_dilation, binary_fill_holes
#from skimage.measure import label, regionprops
#from skimage.morphology import square

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
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figure size as necessary
    cax = ax.imshow(occupancy_grid, cmap='Greys', origin='lower')
    ax.set_title('Occupancy Grid')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, occupancy_grid.shape[1]])
    ax.set_ylim([0, occupancy_grid.shape[0]])
    ax.grid(True)

    plt.show()

def print_occupancy_grid(occupancy_grid):
    """
    Prints the occupancy grid to the console.

    Args:
    - occupancy_grid (np.ndarray): The occupancy grid to print.
    """
    for row in occupancy_grid:
        print(' '.join('1' if cell else '0' for cell in row))


def save_occupancy_grid_to_csv(occupancy_grid, output_file):
    pd.DataFrame(occupancy_grid).to_csv(output_file, index=False, header=False)


def load_occupancy_grid(file_path):
    """
    Loads an occupancy grid from a CSV file.

    Args:
    - file_path (str): Path to the CSV file containing the occupancy grid.

    Returns:
    - occupancy_grid (np.ndarray): The loaded occupancy grid as a Numpy array.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)
    
    # Convert the DataFrame to a Numpy array
    occupancy_grid = df.to_numpy()
    
    return occupancy_grid

"""def identify_and_preprocess_obstacles(grid):
    """"""
    Identify edges and preprocess the grid to fill obstacles.

    Args:
    - grid (np.ndarray): The initial occupancy grid.

    Returns:
    - preprocessed_grid (np.ndarray): The occupancy grid with obstacles filled.
    """"""
    # Dilation to close small gaps and ensure borders are continuous
    dilated_grid = binary_dilation(grid, structure=square(3))

    # Filling the obstacles
    filled_grid = binary_fill_holes(dilated_grid)


        # Step 1: Set borders as occupied
    filled_grid[0, :] = 1  # Top border
    filled_grid[-1, :] = 1  # Bottom border
    filled_grid[:, 0] = 1  # Left border
    filled_grid[:, -1] = 1  # Right border

    return filled_grid.astype(int)"""

# Usage example
updated_file_path = 'C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map_updated.csv'
#occupancy_grid = load_occupancy_grid(updated_file_path)

# Usage example
file_path = 'C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map.csv'
#map_size = 3.5  # meters
#resolution = 0.001  # meter per cell

#occupancy_grid = create_occupancy_grid(file_path, map_size, resolution)
#preprocessed_grid = identify_and_preprocess_obstacles(occupancy_grid)
#visualize_occupancy_grid(preprocessed_grid)
#save_occupancy_grid_to_csv(preprocessed_grid, "C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map_updated.csv")

