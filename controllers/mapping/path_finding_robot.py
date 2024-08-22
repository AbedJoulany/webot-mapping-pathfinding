import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController
from DWA.dwa_controller import DWA
from safe_interval_manager import SafeIntervalManager
from visualize_grid import create_occupancy_grid, load_occupancy_grid
import csv
import os
import threading

SAFE_DISTANCE_THRESHOLD = 0.25 # Example: 0.2 meters

class PathFindingRobotController(BaseRobotController):
    _instance = None
    
    def __init__(self, robot_name="e-puck"):
        super().__init__(robot_name)
        self._initialize_path_planning()

    def _initialize_path_planning(self):
        file_path = 'map.csv'
        self.occupancy_grid = load_occupancy_grid('C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map_updated.csv')
        #self.occupancy_grid = create_occupancy_grid(file_path, self.map_size[0], self.resolution)
        self.pathfinder = AStarPathfinder(self.occupancy_grid, self.resolution, self.map_size, self.distance_between_wheels)
        self.data_file_path = 'robot_movement_data.csv'  # Define the path for the output CSV file
        self.data_file = None
        self.data_writer = None
        self.open_data_file()
        self.obstacle_positions = []
        self.obstacle_positions = load_obstacle_positions("C:/Users/abeda/webot-mapping-pathfinding/controllers/e_puck_controller/simulation_data.csv")
        self.future_obstacle_positions = []

        self.safe_interval_manager = SafeIntervalManager(self.map_size, self.resolution)
        self.pathfinder.safe_interval_manager = self.safe_interval_manager
        self.found_path = None  # Shared variable to store the path

        self.obstacle = self._robot.getFromDef("E-PUCK-OBS")

    def find_path_in_thread(self, start_position, goal_position):
        # Pathfinding runs in a separate thread
        self.found_path = self.pathfinder.find_path(start_position, goal_position)


    def plan_and_follow_path_dwa(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return

        config = {
            'max_speed': 0.25,
            'min_speed': -0.25,
            'max_yaw_rate': 6.28,
            'max_accel': 0.125,
            'max_delta_yaw_rate': 3.14,
            'velocity_resolution': 0.01,
            'yaw_rate_resolution': np.radians(1.0),
            'predict_time': 3.0,
            'to_goal_cost_gain': 2.0,  # Increased to prioritize moving towards the goal
            'obstacle_cost_gain': 0.5,  # Decreased to allow the robot to prioritize the goal more
            'speed_cost_gain': 1.0,
            'dt': 0.064
        }

        dwa = DWA(config)
        # Create and start the thread for pathfinding
        pathfinding_thread = threading.Thread(target=self.find_path_in_thread, args=(start_position, goal_position))
        pathfinding_thread.start()

        # Wait for the pathfinding thread to complete
        pathfinding_thread.join()

        self.found_path = simplify_path_by_angle(self.found_path, angle_threshold=170)  # Adjust threshold as needed
        self.visualize_path(self.found_path)

        while self.robot.step(self._timestep) != -1:
            current_position  = self.get_robot_pose_from_webots()
            current_orientation = self._epuck_robot.getOrientation()[2]  # Assuming 2D navigation, yaw only
            if self.found_path:
                goal = self.found_path[0]  # Next waypoint in path
            else:
                print("No valid path found.")
                return


            #current_speed = (self._left_motor.getVelocity() + self._right_motor.getVelocity()) * self.wheel_radius / 2
            current_speed = math.sqrt(self._epuck_robot.getVelocity()[0]**2+self._epuck_robot.getVelocity()[2]**2)
            current_yaw_rate = (self._right_motor.getVelocity() - self._left_motor.getVelocity()) * self.wheel_radius / self.distance_between_wheels
            #current_yaw_rate = self._epuck_robot.getVelocity()[4]
            current_state = [current_position[0], current_position[1], current_yaw_rate, current_speed, current_position[2]]  # [x, y, yaw, v, omega]

            # Assume you have a function to get obstacles from LIDAR or other sensors
            obstacles = [self.obstacle.getPosition()[:2]]

            best_u, best_trajectory = dwa.dwa_control(current_state, goal, obstacles)

            # Extract linear and angular velocities
            v = best_u[0]
            w = best_u[1]
            #print(best_u)
            # Compute the wheel speeds based on differential drive equations
            left_wheel_speed = (v - w * self.distance_between_wheels / 2) / self.wheel_radius
            right_wheel_speed = (v + w * self.distance_between_wheels / 2) / self.wheel_radius

            self.set_motor_speeds(left_wheel_speed, right_wheel_speed)
            # Check if waypoint is reached
            if np.hypot(goal[0] - current_position[0], goal[1] - current_position[1]) < 0.1:
                self.found_path.pop(0)  # Remove waypoint from path if reached


    def plan_and_follow_path(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return

        goal_reached = False
        self.robot.step(self._timestep)
        while not goal_reached:
            start_position = self.get_robot_pose_from_webots()

            # Create and start the thread for pathfinding
            pathfinding_thread = threading.Thread(target=self.find_path_in_thread, args=(start_position, goal_position))
            pathfinding_thread.start()

            # Wait for the pathfinding thread to complete
            pathfinding_thread.join()
            #self.found_path = self.pathfinder.find_path(start_position, goal_position)
            if self.found_path is None:
                print("No valid path found.")
                return

            self.found_path = simplify_path_by_angle(self.found_path, angle_threshold=170)  # Adjust threshold as needed
            self.visualize_path(self.found_path)

            waypoints = [self.pathfinder.grid_to_world(grid_position) for grid_position in self.found_path]
            goal_reached = self.move_robot_to_waypoints(waypoints)

    def move_robot_to_waypoints(self, waypoints):
        current_position = self.get_robot_pose_from_webots()
        for waypoint in waypoints:
            if self._move_towards_waypoint(current_position, waypoint):
                current_position = self.get_robot_pose_from_webots()
            else:
                return False

        self.set_motor_speeds(0, 0)
        return True

    def _move_towards_waypoint(self, current_position, target_position):
        def reached_waypoint(curr_pos, target_pos, threshold=0.2):
            return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

        while not reached_waypoint(current_position, target_position):
            self.update_pid(current_position, target_position)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()

            # Check for potential future collisions
            velocity = self._epuck_robot.getVelocity()
            linear_velocity = velocity[:3]
            angular_velocity = velocity[3:]
            """if self.check_for_future_collisions(current_position, target_position, self.obstacle_positions, linear_velocity):
                print("Potential collision detected in future steps! Stopping or rerouting...")
                self.set_motor_speeds(0, 0)

                return False"""

            #self.data_writer.writerow([self._robot.getTime(), current_position[0], current_position[1], current_position[2], target_position])
        return True

    def check_for_future_collisions(self, current_position, target_position, obstacle_positions, velocity, steps_ahead=3):
        future_robot_positions = predict_future_positions(current_position, target_position, velocity, steps_ahead)
        current_time_step = int((self.robot.getTime() * 100) / self._timestep)
        self.future_obstacle_positions = get_future_obstacle_positions(obstacle_positions, current_time_step, steps_ahead+3)

        for i in  range(min(len(future_robot_positions),len(self.future_obstacle_positions))):
                # Check if the obstacle is close to the robot's future position
            distance = np.linalg.norm(np.array(future_robot_positions[i][:2]) - np.array(self.future_obstacle_positions[i][:2]))
            if distance < SAFE_DISTANCE_THRESHOLD:
                    # Now check if the obstacle is in the robot's path
                if self.is_obstacle_in_robot_path(current_position, target_position, self.future_obstacle_positions[i]):
                    return True
        return False

    def is_obstacle_in_robot_path(self, current_position, target_position, obstacle_position):
        """
        Check if an obstacle lies in the robot's path by projecting its position onto the line segment between
        current_position and target_position.
        """
        robot_pos = np.array(current_position[:2])
        target_pos = np.array(target_position[:2])
        obstacle_pos = np.array(obstacle_position[:2])

        # Vector from robot to target
        robot_to_target = target_pos - robot_pos
        robot_to_obstacle = obstacle_pos - robot_pos

        # Project the obstacle onto the robot's path (line segment)
        projection = np.dot(robot_to_obstacle, robot_to_target) / np.dot(robot_to_target, robot_to_target)
        # Check if the projection lies between the robot and the target (i.e., between 0 and 1)
        if 0 <= projection <= 1:
            # Calculate the closest point on the line segment to the obstacle
            closest_point_on_path = robot_pos + projection * robot_to_target
            # Calculate the distance from the obstacle to the closest point on the path
            distance_to_path = np.linalg.norm(obstacle_pos - closest_point_on_path)

            # Return True if the obstacle is within the safe distance threshold from the path
            if distance_to_path < SAFE_DISTANCE_THRESHOLD:
                return True

        return False



    def visualize_path(self, path):
        grid = np.array(self.pathfinder.occupancy_grid)
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figure size as necessary
        ax.imshow(grid, cmap='Greys', origin='lower')  # Display the grid

        # Extracting X and Y coordinates from the path
        x_coords, y_coords,z = zip(*path)
        ax.plot(x_coords, y_coords, marker='o', color='red')  # Plot the path

        ax.set_xlim([0, grid.shape[1]])
        ax.set_ylim([0, grid.shape[0]])
        ax.set_title("Occupancy Grid and Path")
        plt.show()

    def open_data_file(self):
        # Open the file in append mode and create a CSV writer
        self.data_file = open(self.data_file_path, 'a', newline='')
        self.data_writer = csv.writer(self.data_file)
        # Write the header if the file is empty
        if os.stat(self.data_file_path).st_size == 0:
            self.data_writer.writerow(['Time Step', 'X', 'Y', 'Z', 'Current Waypoint'])

    def close_data_file(self):
        if self.data_file:
            self.data_file.close()

    def __del__(self):
        # Ensure the file is closed when the controller is destroyed
        self.close_data_file()


def simplify_path_by_angle(path, angle_threshold=170):
    """
    Simplifies a path by removing unnecessary waypoints, keeping those that represent significant turns.
    Args:
        path (list of tuples): The original path as a list of (x, y) tuples.
        angle_threshold (float): The minimum angle in degrees to consider a turn significant.

    Returns:
        list of tuples: The simplified path.
    """
    if len(path) < 3:  # No simplification possible if there are less than three points
        return path

    simplified_path = [path[0]]  # Always keep the starting point

    def calculate_angle(p1, p2, p3):
        """Calculate the angle between three points, p1-p2-p3, where p2 is the vertex."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine_angle)  # radians
        return np.degrees(angle)  # convert to degrees

    for i in range(1, len(path) - 1):
        angle = calculate_angle(path[i - 1], path[i], path[i + 1])
        if angle < angle_threshold:
            simplified_path.append(path[i])

    simplified_path.append(path[-1])  # Always keep the end point

    return simplified_path


def load_obstacle_positions(file_path):
    obstacle_positions = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_step = float(row['Time'])
            x = float(row['Obstacle_X'])
            y = float(row['Obstacle_Y'])
            z = float(row['Obstacle_Theta'])
            obstacle_positions.append((time_step, (x, y, z)))
    return obstacle_positions


def predict_future_positions(current_position, target_position, velocity, steps_ahead):
    future_positions = []
    for i in range(1, steps_ahead + 1):
        direction_vector = np.array(target_position) - np.array(current_position)
        unit_vector = direction_vector / np.linalg.norm(direction_vector)
        future_position = np.array(current_position) + (unit_vector * velocity * i)
        future_positions.append(tuple(future_position))
    return future_positions


def get_future_obstacle_positions(obstacle_positions, current_time_step, steps_ahead):
    future_positions = []
    for i in range(current_time_step, current_time_step + steps_ahead):
        future_positions.extend(
            [pos for time_step, pos in obstacle_positions if round(time_step,3) == round((i*64/100),3)]
        )
    return future_positions




