import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController
from EKFS.ekf2 import EKF

class PathFindingRobotEKFController(BaseRobotController):
    _instance = None



    def __init__(self, robot_name="e-puck"):
        """
        if not hasattr(self, '_initialized'):  # Avoid reinitialization

        """
        super().__init__(robot_name)
        self._initialize_path_planning()
        self._initialized = True
        self.ekf = self.initialize_ekf()
        self.obstacle = self._robot.getFromDef("E-PUCK-OBS")
        self.rotation_field = self.obstacle.getField("rotation")
        self.previous_obstacle_position = self.get_obstacle_position()

    def initialize_ekf(self):
        dt = self._timestep / 64.0  # Convert timestep to seconds
        state_dim = 3  # [x, y, theta]
        meas_dim = 3   # [x, y, theta]
        control_dim = 2  # [v, omega]

        ekf = EKF(dt, state_dim, meas_dim, control_dim)

        F = np.eye(state_dim)
        ekf.set_F(F)

        H = np.eye(meas_dim, state_dim)
        ekf.set_H(H)

        q = 0.1  # Process noise scalar
        Q = q * np.eye(state_dim)
        ekf.set_Q(Q)

        r = 0.5  # Measurement noise scalar
        R = r * np.eye(meas_dim)
        ekf.set_R(R)

        return ekf

    def _initialize_path_planning(self):
        self.occupancy_grid = [[0] * int(self.map_size[0] / self.resolution) for _ in range(int(self.map_size[1] / self.resolution))]
        self.pathfinder = AStarPathfinder(self.occupancy_grid, self.resolution, self.map_size)

    def plan_and_follow_path(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return
        print(f'goal_position = {goal_position}')
        self.goal_position = goal_position  # Set the goal_position
        self.robot.step(self._timestep)
        count = 0
        while count < 1:
            path = self.pathfinder.find_path(start_position, goal_position)
            if path is None:
                print("No valid path found.")
                return

            waypoints = [self.pathfinder.grid_to_world(grid_position) for grid_position in path]
            # Predict future positions of the obstacle
            future_obstacle_positions = self.predict_future_obstacle_positions(steps= 20)
            """len(waypoints)"""
            print(f"future_obstacle_positions = {future_obstacle_positions}")
            collision_detected = False
            for robot_pos, obstacle_pos in zip(waypoints, future_obstacle_positions):
                if self.check_collision(robot_pos, obstacle_pos, collision_radius=0.5):  # If the distance is less than 0.5 meters
                    print("Collision predicted. Replanning path...")
                    collision_detected = True
                    break
            if collision_detected:
                print("collision_detected")

            if not collision_detected:
                print("no collision detected")
                break
            count+=1
        self.move_robot_to_waypoints(waypoints)


    def _move_towards_waypoint(self, current_position, target_position):
        def reached_waypoint(curr_pos, target_pos, threshold=0.05):
            return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

        while not reached_waypoint(current_position, target_position):
            self.update_pid(current_position, target_position)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()


    def get_obstacle_position(self):
        target_pos = self.obstacle.getPosition()
        robot_rot = self.rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)

        return [round(target_pos[0], 3), round(target_pos[1], 3), robot_rot_z]

    def get_obstacle_actual_velocity(self):
        velocity = self.obstacle.getVelocity()
        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]

        return np.array([np.linalg.norm(linear_velocity), np.linalg.norm(angular_velocity)])
    

    def predict_future_obstacle_positions(self, steps=10):
        future_positions = []
        for _ in range(steps):
            obstacle_velocity = self.get_obstacle_actual_velocity()
            control_input = np.array(obstacle_velocity)
            self.ekf.predict(control_input)
            predicted_position = self.ekf.x
            future_positions.append(predicted_position[0])
        return future_positions

    def check_collision(self, robot_pos, obstacle_pos, collision_radius):
        dx = robot_pos[0] - obstacle_pos[0]
        dy = robot_pos[1] - obstacle_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < collision_radius
