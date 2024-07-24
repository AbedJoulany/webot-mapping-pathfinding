import math
import numpy as np
import cv2
import csv

from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

class DataCollectorRobotController(BaseRobotController):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):  # Avoid reinitialization
            super().__init__()
            self._initialized = True

            self.obstacle = self._robot.getFromDef("FourWheelsRobot")

            self.data = []
            self.header = ['Time', 'Estimated_X', 'Estimated_Y', 'Estimated_Velocity', 
                           'Actual_X', 'Actual_Y', 'Actual_Velocity']
            with open('simulation_data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)  # Write header
            self.robot_pose_encoder = self.get_robot_pose_from_webots()
            self.previous_positions = []
            self.previous_times = []
            #self.ekf = self.initialize_ekf()
    """
    def initialize_ekf(self):
        dt = self._timestep / 1000.0  # Convert timestep to seconds
        state_dim = 3  # [x, y, theta]
        meas_dim = 3   # [x, y, theta]
        control_dim = 2  # [v, omega]


        ekf = EKF(dt, state_dim, meas_dim, control_dim)


        # State transition model (linearized)
        F = np.eye(state_dim)

        ekf.set_F(F)

        # Measurement model (linearized)
        H = np.eye(meas_dim, state_dim)

        ekf.set_H(H)

        # Process noise covariance
        q = 0.1  # Process noise scalar
        Q = q * np.eye(state_dim)
        ekf.set_Q(Q)

        # Measurement noise covariance
        r = 0.5  # Measurement noise scalar
        R = r * np.eye(meas_dim)
        ekf.set_R(R)

        return ekf
    """

    def get_obstacle_position(self):
        target_pos = self.obstacle.getPosition()
        return np.array([round(target_pos[0], 3), round(target_pos[1], 3)])

    def get_obstacle_velocity():

        return 2

    def move_robot_to_waypoints(self, waypoints):
        current_position = self.get_robot_pose_from_webots()
        for waypoint in waypoints:
            #print(waypoint)
            self._move_towards_waypoint(current_position, waypoint)
            current_position = self.get_robot_pose_from_webots()
        self.set_motor_speeds(0, 0)
        return True

    def _move_towards_waypoint(self, current_position, target_position):
        def reached_waypoint(curr_pos, target_pos, threshold=0.2):
            return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

        self.adjust_heading(current_position, target_position)

        while not reached_waypoint(current_position, target_position):

            self.update_pid(current_position, target_position)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()
            self.collect_data()

    def update_pid(self, current_position, target_position):
        current_orientation = current_position[2]

        distance_error, heading_error = self.calculate_control_signal(target_position, current_position, current_orientation)

        dt = self.timestep / 64
        linear_velocity = self.distance_pid.update(distance_error, dt)
        angular_velocity = self.heading_pid.update(heading_error, dt)
        
        left_speed = self.limit_speed(linear_velocity - angular_velocity, -6.28, 6.28)
        right_speed = self.limit_speed(linear_velocity + angular_velocity, -6.28, 6.28)
        
        self.set_motor_speeds(left_speed, right_speed)

    def calculate_error(self, target, current):
        return math.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

    def calculate_control_signal(self, target, current_position, current_orientation):
        angle_to_target = math.degrees(math.atan2(target[1] - current_position[1], target[0] - current_position[0]))
        heading_error = angle_to_target - math.degrees(current_orientation)

        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        distance_error = self.calculate_error(target, current_position)

        return distance_error, heading_error

    def limit_speed(self, speed, min_speed, max_speed):
        return max(min_speed, min(speed, max_speed))


    def move_random(self):
        waypoints = self.generate_random_waypoints(num_waypoints=100)
        self.move_robot_to_waypoints(waypoints)

    def generate_random_waypoints(self, num_waypoints):
        waypoints = []
        for _ in range(num_waypoints):
            x = round(np.random.uniform(-1.0, 1.0), 3)
            y = round(np.random.uniform(-1.0, 1.0), 3)
            waypoints.append((x, y))
        return waypoints

    def collect_data(self):
        def is_obstacle_out_of_lidar_range(lidar_data):
            return all(distance == float('inf') for distance in lidar_data)
        lidar_data = self.lidar.getRangeImage()
        if is_obstacle_out_of_lidar_range(lidar_data):
            self.previous_positions = []
            self.previous_times = []
            self.history = []
            return

        left_speed = self.left_motor.getVelocity()
        right_speed = self.right_motor.getVelocity()
        speed = (left_speed + right_speed) / 2
        current_time = self.robot.getTime()

        obstacle_velocity = self.get_obstacle_actual_velocity()
        print(f"obstacle_velocity = {np.linalg.norm(obstacle_velocity)}")

        v, w = self.odometry()
        z, z_points, angle_i, pointCloud = self.get_lidar_points()
        obstacle_positions, obstacle_orientations = self.estimate_obstacle_position_and_orientation(z_points)
        estimated_velocity = self.estimate_obstacle_velocity(obstacle_positions, current_time)
        print(f"estimated_velocity = {estimated_velocity}")


        #for pos, ori in zip(obstacle_positions, obstacle_orientations):
        #    print(f"Obstacle Position: {pos}, Orientation: {ori}")


    def get_obstacle_speed(self):
        lidar_data = self.lidar.getRangeImage()
        if not lidar_data:  # Check if lidar data is empty
            return 0.0
        else:
            return 1

    def estimate_obstacle_position_and_orientation(self, z_points):
        if len(z_points) == 0:
            return [], []

        points = np.array(z_points)
        
        # Clustering the Lidar points to identify individual obstacles
        clustering = DBSCAN(eps=0.1, min_samples=5).fit(points)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        obstacle_positions = []
        obstacle_orientations = []
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            
            cluster_points = points[labels == label]
            
            # Estimate the position (centroid of the cluster)
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate the orientation based on the change in x and y positions
            dx = cluster_points[-1, 0] - cluster_points[0, 0]
            dy = cluster_points[-1, 1] - cluster_points[0, 1]
            orientation = np.arctan2(dy, dx)
            
            obstacle_positions.append(centroid)
            obstacle_orientations.append(orientation)
        
        return obstacle_positions, obstacle_orientations

    def save_collected_data(self):
        with open('simulation_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

    def get_obstacle_actual_velocity(self):
        velocity = self.obstacle.getVelocity()
        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]

        return np.array([np.linalg.norm(linear_velocity), np.linalg.norm(angular_velocity)])
    


    def estimate_obstacle_velocity(self, current_positions, current_time):
        if len(current_positions) == 0:
            return 0.0  # No positions to estimate velocity

        if not hasattr(self, 'history'):
            self.history = []

        for current_position in current_positions:
            # Add the current position and time to the history
            self.history.append((current_position, current_time))
        
        if len(self.history) < 2:
            return 0.0  # Not enough data to estimate velocity

        # Calculate velocities between each pair of consecutive positions
        velocities = []
        for i in range(1, len(self.history)):
            previous_position, previous_time = self.history[i - 1]
            current_position, current_time = self.history[i]
            
            # Calculate the time difference
            time_diff = current_time - previous_time
            time_diff = round(time_diff, 3)
            if time_diff == 0:
                velocities.append(np.array([0.0, 0.0]))  # Avoid division by zero
                continue

            # Calculate the difference in position
            position_diff = abs(current_position - previous_position)

            # Calculate velocity
            velocity = position_diff / time_diff
            velocities.append(velocity)

        # Update history to keep only the last few positions
        if len(self.history) > 5:  # Limit history size to 5 entries
            self.history.pop(0)

        # Calculate the average velocity vector
        avg_velocity = np.mean(velocities, axis=0)

        # Return the magnitude of the average velocity vector
        return np.linalg.norm(avg_velocity)

