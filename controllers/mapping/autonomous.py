import numpy as np
import cv2
from robot_controller import RobotController
from map_update import calculate_grid_coordinates, is_within_grid
import numpy as np

params = {
    "coef": 0.6,
    "max_speed": 0.1 * 0.6,
    "wheelRadius": 0.0205,
    "axleLength": 0.0568,
    "updateFreq": 1,  # Timesteps between odometry updates
    "plotFreq": 50,  # Timesteps between map visualizations (optional)
}

def move_robot(robot, velocity, rotation):
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setVelocity(velocity + rotation)
    right_motor.setVelocity(velocity - rotation)

def autonomous_navigation(robot):

    robot = controller.robot
    timestep = controller.timestep
    lidar = controller.lidar
    left_motor = controller.left_motor
    right_motor = controller.right_motor
    gps = controller.gps
    compass = controller.compass
    keyboard = controller.keyboard
    
    grid_size = (50, 50)
    grid_resolution = 0.1
    grid_offset = (grid_size[0] // 2, grid_size[1] // 2)  # (25, 25) for a 50x50 grid
    occupancy_grid = np.zeros(grid_size)

    num_particles = 100
    initial_position = (2.0, 3.0, 0.0)
    particles = initialize_particles(num_particles, initial_position)

    velocity = 1.0
    rotation = 0.0
    obstacle_threshold = 0.3

    while robot.step(timestep) != -1:
        lidar_data = lidar.getRangeImage()

        if lidar_data is None:
            #print("Warning: Lidar data is None. Skipping this timestep.")
            continue

        lidar_resolution = len(lidar_data)

        particles = [motion_model(p, robot, timestep) for p in particles]
        particles = [sensor_model(p, lidar_data, occupancy_grid, lidar.getFov(), lidar_resolution, grid_size, grid_resolution) for p in particles]
        particles = resample(particles)
        occupancy_grid = update_map(occupancy_grid, particles, lidar_data, lidar.getFov(), lidar_resolution, grid_size, grid_resolution)

        min_distance = min(lidar_data)
        if min_distance < obstacle_threshold:
            rotation = np.random.choice([-1.0, 1.0]) * 0.5

        move_robot(robot, velocity, rotation)

        # Optional visualization
        if timestep % params["plotFreq"] == 0:
            visualize_grid(occupancy_grid)

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




def update_robot_position(robot_position, velocity, rotation):
    x, y, theta = robot_position
    # Update position based on velocity and rotation
    x += velocity * np.cos(theta)
    y += velocity * np.sin(theta)
    theta += rotation
    return x, y, theta

            
def update_occupancy_grid(occupancy_grid, robot_position, grid_resolution):
    x, y, _ = robot_position
    grid_x = int(x / grid_resolution)
    grid_y = int(y / grid_resolution)
    if 0 <= grid_x < occupancy_grid.shape[0] and 0 <= grid_y < occupancy_grid.shape[1]:
        occupancy_grid[grid_x][grid_y] += 1  # Example update method (you can adjust this based on your application)
    return occupancy_grid
