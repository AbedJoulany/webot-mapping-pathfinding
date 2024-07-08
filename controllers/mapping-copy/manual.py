import numpy as np
import cv2
from robot_controller import RobotController
from map_update import calculate_grid_coordinates, is_within_grid, convert_lidar_range_to_coordinates, coords_to_grid


def manual_navigation(controller):
    robot = controller.robot
    timestep = controller.timestep
    lidar = controller.lidar
    left_motor = controller.left_motor
    right_motor = controller.right_motor
    gps = controller.gps
    compass = controller.compass
    keyboard = controller.keyboard

    velocity = 0.0
    rotation = 0.0

    grid_size = (100, 100)
    grid_resolution = 0.1
    grid_offset = (grid_size[0] // 2, grid_size[1] // 2)
    occupancy_grid = np.zeros(grid_size)

    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        if key == keyboard.UP:
            velocity = 1.0
        elif key == keyboard.DOWN:
            velocity = -1.0
        else:
            velocity = 0.0

        if key == keyboard.LEFT:
            rotation = -1.0
        elif key == keyboard.RIGHT:
            rotation = 1.0
        else:
            rotation = 0.0

        move_robot(robot, velocity * 6.28, rotation)

        # Update robot position and occupancy grid based on GPS and lidar
        alpha = np.arctan2(compass.getValues()[0], compass.getValues()[1])
        pos = gps.getValues()
        robot_position = (pos[0], pos[1], alpha)

        lidar_data = lidar.getRangeImage()
        if lidar_data is not None:
            update_occupancy_grid_manual(occupancy_grid, lidar, robot_position, lidar_data, grid_resolution, grid_offset, controller)

        visualize_grid(occupancy_grid)

def update_occupancy_grid_manual(occupancy_grid, lidar, robot_position, lidar_data, grid_resolution, grid_offset, controller):
    robot_x, robot_y, robot_theta = robot_position

    lidar_fov = lidar.getFov()
    num_measurements = len(lidar_data)
    
    grid_size = (100, 100)


    for i in range(num_measurements):
        distance = lidar_data[i]
        angle = i * (lidar_fov / num_measurements) - (lidar_fov / 2)

        if np.isnan(distance) or np.isnan(angle):
            continue  # Skip this measurement if it's invalid

        # Calculate local coordinates of the obstacle detected by lidar
        local_x = distance * np.cos(angle)
        local_y = distance * np.sin(angle)

        if np.isnan(local_x) or np.isnan(local_y):
            continue  # Skip if local coordinates are invalid

        # Rotate local coordinates by -robot_theta (to the left)
        rotated_local_x = local_x * np.cos(-robot_theta) - local_y * np.sin(-robot_theta) 
        rotated_local_y = local_x * np.sin(-robot_theta) + local_y * np.cos(-robot_theta)

        # Calculate global coordinates by subtracting robot position
        global_x = robot_x + rotated_local_x
        global_y = robot_y + rotated_local_y
        
        print(robot_x, robot_y)

        if np.isnan(global_x) or np.isnan(global_y):
            print(f"Skipping invalid global coordinates: global_x={global_x}, global_y={global_y}")
            continue  # Skip updating the grid if global coordinates are invalid

        # Convert global coordinates to grid coordinates using world2map function
        #map_x, map_y = controller.world2map((global_x, global_y, 0), grid_size, grid_resolution)

               # Convert global coordinates to grid coordinates
        try:
            grid_x = int((global_x + grid_resolution * grid_offset[0]) / grid_resolution)
            grid_y = int((global_y + grid_resolution * grid_offset[1]) / grid_resolution)
            #print(grid_x, grid_y)
        except ValueError as e:
            print(f"Skipping invalid grid coordinates: global_x={global_x}, global_y={global_y}")
            continue
        
        # Update occupancy grid if the grid coordinates are within bounds
        #if is_within_grid(map_x, map_y, occupancy_grid):
        #    occupancy_grid[map_x, map_y] += 1  # Increment occupancy value
            
        if is_within_grid(grid_x, grid_y, occupancy_grid):
            occupancy_grid[-grid_x][-grid_y] += 1  # Increment occupancy value

    return occupancy_grid

# Keep the rest of the code (move_robot, visualize_grid, etc.) unchanged


"""

coords = convert_lidar_range_to_coordinates(lidar_data, lidar_fov, lidar.getHorizontalResolution())
    
    map_grid_size = int(10 / grid_resolution)    
    i = 0
    for coord in coords:
        grid_x, grid_y = coords_to_grid(coord, 10, grid_resolution)
        if 0 <= grid_x < map_grid_size and 0 <= grid_y < map_grid_size:
         
            if lidar_data[i] < 0.5 and is_within_grid(grid_y, grid_x, occupancy_grid):
                # Set obstacle pixel to OBSTACLE_COLOR
                occupancy_grid[grid_y, grid_x] = 0
            else:
                # Set free space pixel to FREE_SPACE_COLOR
                occupancy_grid[grid_y, grid_x] = 255
        i+=1

            
"""

def move_robot(robot, velocity, rotation):
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setVelocity(velocity + rotation)
    right_motor.setVelocity(velocity - rotation)


def visualize_grid(occupancy_grid):
    # Scale factor for visualization
    scale = 10

    # Create an empty image
    image_size = (occupancy_grid.shape[0] * scale, occupancy_grid.shape[1] * scale)
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Draw occupancy grid
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            color = int(255 - occupancy_grid[x][y] * 2.55)  # Convert occupancy value to grayscale
            cv2.rectangle(image, (x * scale, y * scale), ((x + 1) * scale - 1, (y + 1) * scale - 1), (color, color, color), -1)

    # Display image
    cv2.imshow("Occupancy Grid", image)
    cv2.waitKey(1)  # Adjust as needed for display time



