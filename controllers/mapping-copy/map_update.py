import numpy as np
import math

def calculate_grid_coordinates(distance, angle, robot_position, grid_size, grid_resolution, grid_offset):
    robot_x, robot_y, robot_theta = robot_position

    if np.isnan(distance) or np.isinf(distance) or np.isnan(angle):
        return None, None

    local_x = distance * np.cos(angle)
    local_y = distance * np.sin(angle)

    global_x = local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta) + robot_x
    global_y = local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta) + robot_y

    if np.isnan(global_x) or np.isinf(global_x) or np.isnan(global_y) or np.isinf(global_y):
        return None, None

    # Map global coordinates to grid coordinates with offset
    grid_x = int(global_x / grid_resolution) + grid_offset[0]
    grid_y = int(global_y / grid_resolution) + grid_offset[1]

    return grid_x, grid_y


def is_within_grid(x, y, grid):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
    
    

def update_map(occupancy_grid, particles, sensor_data, lidar_fov, lidar_resolution, grid_size, grid_resolution):
    for particle in particles:
        for i in range(len(sensor_data)):
            distance = sensor_data[i]
            angle = i * (lidar_fov / len(sensor_data)) - (lidar_fov / 2)
            x, y = calculate_grid_coordinates(distance, angle, (particle.x, particle.y, particle.theta), grid_size, grid_resolution)
            if x is not None and y is not None and is_within_grid(x, y, occupancy_grid):
                # Update occupancy grid based on particle weight (higher weight indicates higher confidence)
                occupancy_grid[x][y] += particle.weight
    return occupancy_grid
    
    
    

def convert_lidar_range_to_coordinates(lidar_data, lidar_fov, horizontal_res):

    angle_increment = lidar_fov / (horizontal_res - 1)
    ranges = lidar_data
    
    
    # Convert LIDAR range data to (x,y) coordinates
    coords = []
    for i in range(horizontal_res):
        angle = -lidar_fov / 2 + angle_increment * i
        distance = ranges[i]
        if distance > 0:
            x = distance * math.sin(angle)
            y = distance * math.cos(angle)

            if (not math.isnan(x) and not math.isnan(y)):
                if str(x) == "inf" or str(y) == "inf" or str(x) == "-inf" or str(y) == "-inf":
                    continue
                coords.append((x, y))
    
    return coords
    
    
# Define helper function to convert coordinates to grid indices
def coords_to_grid(coords, grid_size, grid_resolution):
    grid_x = int((coords[0] + grid_size / 2) / grid_resolution) + 10
    grid_y = int((coords[1] + grid_size / 2) / grid_resolution) + 10
    return grid_x, grid_y
