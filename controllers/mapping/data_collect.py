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
            self.header = ['Time', 'Obstacle_X', 'Obstacle_Y', 'Obstacle_Theta']

            with open('simulation_data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)  # Write header
            self.robot_pose_encoder = self.get_robot_pose_from_webots()
            self.previous_positions = {}
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

    def get_obstacle_position():
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
            self.set_motor_speeds(6.28, 6.28)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()
            self.collect_data()

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
        lidar_data = self.lidar.getRangeImage()
        if not lidar_data:  # Check if lidar data is empty
            lidar_data = [float('inf')] * self.lidar.getHorizontalResolution()  # Use 'inf' as placeholder

        left_speed = self.left_motor.getVelocity()
        right_speed = self.right_motor.getVelocity()
        speed = (left_speed + right_speed) / 2

        try:
            obstacle_speed = self.get_obstacle_speed()
        except ValueError:
            obstacle_speed = 0.0  # Default value in case of error

        try:
            obstacle_distance = min(lidar_data) if lidar_data else float('inf')
        except ValueError:
            obstacle_distance = float('inf')

        v, w = self.odometry()
        z, z_points, angle_i, pointCloud = self.get_lidar_points()
        obstacle_positions, obstacle_orientations = self.estimate_obstacle_position_and_orientation(z_points)

        for pos, ori in zip(obstacle_positions, obstacle_orientations):
            print(f"Obstacle Position: {pos}, Orientation: {ori}")


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
            
            obstacle_positions.append(centroid)
            obstacle_orientations.append(orientation)
        
        return obstacle_positions, obstacle_orientations

    def save_collected_data(self):
        pass

    def get_obstacle_distances(self):
        # Placeholder method to obtain the distances to obstacles
        # You need to implement this based on your simulation environment
        return self.lidar.getRangeImage()  # Replace with actual implementation

