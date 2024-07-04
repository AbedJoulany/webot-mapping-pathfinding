from controller import Robot, GPS, InertialUnit, DistanceSensor
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# Constants
TIME_STEP = 64
GRID_SIZE = 50  # size of the grid
GRID_RESOLUTION = 0.1  # grid resolution in meters (each cell represents 10cm x 10cm)
MAP_SIZE = GRID_SIZE * GRID_RESOLUTION  # size of the map in meters
MAX_SENSOR_RANGE = 1.0  # max range of the distance sensor in meters

class Mapper:
    def __init__(self):
        self.robot = Robot()

        # Initialize sensors
        self.gps = self.robot.getDevice('global')
        self.gps.enable(TIME_STEP)
        self.imu = self.robot.getDevice('imu')
        self.imu.enable(TIME_STEP)
        self.ds_left = self.robot.getDevice('ds_left')
        self.ds_right = self.robot.getDevice('ds_right')
        self.ds_left.enable(TIME_STEP)
        self.ds_right.enable(TIME_STEP)

        # Initialize motors
        self.motors = []
        motor_names = ['bk_right_wheel', 'fr_right_wheel', 'bk_left_wheel', 'fr_left_wheel']
        for name in motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(float('inf'))  # Set to velocity control
            motor.setVelocity(0.0)
            self.motors.append(motor)
        
        # Map variables
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))  # occupancy grid map
        
    def set_motor_speeds(self, left_speed, right_speed):
        for i in range(2):
            self.motors[i].setVelocity(right_speed)
        for i in range(2, 4):
            self.motors[i].setVelocity(left_speed)

    def get_gps_position(self):
        return self.gps.getValues()

    def get_imu_orientation(self):
        return self.imu.getRollPitchYaw()

    def update_map(self, x, y, sensor_angle, sensor_distance):
        # Convert robot coordinates to grid coordinates
        grid_x = int((x + MAP_SIZE / 2) / GRID_RESOLUTION)
        grid_y = int((y + MAP_SIZE / 2) / GRID_RESOLUTION)

        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            # Calculate obstacle coordinates based on sensor data
            obs_x = x + sensor_distance * math.cos(sensor_angle)
            obs_y = y + sensor_distance * math.sin(sensor_angle)
            
            grid_obs_x = int((obs_x + MAP_SIZE / 2) / GRID_RESOLUTION)
            grid_obs_y = int((obs_y + MAP_SIZE / 2) / GRID_RESOLUTION)
            
            if 0 <= grid_obs_x < GRID_SIZE and 0 <= grid_obs_y < GRID_SIZE:
                self.grid[grid_obs_x, grid_obs_y] = 1  # mark obstacle

    def find_frontiers(self):
        # Detect the boundary between known and unknown areas
        known = self.grid > 0
        unknown = self.grid == 0
        dilated_known = binary_dilation(known, iterations=1)
        frontiers = np.logical_and(dilated_known, unknown)
        return np.argwhere(frontiers)

    def move_towards_goal(self, current_pos, goal_pos):
        goal_x, goal_y = goal_pos
        current_x, current_y, _ = current_pos
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)

        # Get robot orientation (yaw)
        _, _, yaw = self.get_imu_orientation()

        # Calculate the required turn angle
        turn_angle = angle_to_goal - yaw

        # Normalize the angle to the range [-pi, pi]
        while turn_angle > math.pi:
            turn_angle -= 2 * math.pi
        while turn_angle < -math.pi:
            turn_angle += 2 * math.pi

        # Determine motor speeds based on the turn angle
        if abs(turn_angle) > 0.1:
            if turn_angle > 0:
                self.set_motor_speeds(-1.0, 1.0)  # Turn right
            else:
                self.set_motor_speeds(1.0, -1.0)  # Turn left
        else:
            self.set_motor_speeds(1.0, 1.0)  # Move forward

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            # Get sensor readings
            gps_position = self.get_gps_position()
            imu_orientation = self.get_imu_orientation()
            ds_left_value = self.ds_left.getValue()
            ds_right_value = self.ds_right.getValue()

            # Debugging: Print sensor values
            print(f"GPS Position: {gps_position}")
            print(f"IMU Orientation: {imu_orientation}")
            print(f"Left Distance Sensor: {ds_left_value}")
            print(f"Right Distance Sensor: {ds_right_value}")

            # Update localization data
            x, y, z = gps_position
            roll, pitch, yaw = imu_orientation

            # Update map with sensor data
            if ds_left_value < MAX_SENSOR_RANGE:
                self.update_map(x, y, yaw + math.pi/2, ds_left_value)
            if ds_right_value < MAX_SENSOR_RANGE:
                self.update_map(x, y, yaw - math.pi/2, ds_right_value)

            # Find frontiers (boundaries between known and unknown areas)
            frontiers = self.find_frontiers()
            if len(frontiers) > 0:
                # Move towards the closest frontier
                goal = frontiers[0]  # You can implement a better selection strategy here
                goal_pos = (goal[0] * GRID_RESOLUTION - MAP_SIZE / 2, goal[1] * GRID_RESOLUTION - MAP_SIZE / 2)
                self.move_towards_goal((x, y, z), goal_pos)
            else:
                self.set_motor_speeds(0.0, 0.0)  # Stop if no frontiers are found

        # Display the map
        plt.imshow(self.grid, cmap='gray')
        plt.show()

# Create the Mapper instance and run the mapping
mapper = Mapper()
mapper.run()
