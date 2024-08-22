import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController
from visualize_grid import create_occupancy_grid, load_occupancy_grid
import csv
import os
import cvxpy as cp


SAFE_DISTANCE_THRESHOLD = 0.2  # Example: 0.2 meters

class PathFindingRobotMPCController(BaseRobotController):
    _instance = None
    
    def __init__(self, robot_name="e-puck"):
        super().__init__(robot_name)
        self._initialize_path_planning()

    def _initialize_path_planning(self):
        file_path = 'map.csv'
        self.occupancy_grid = load_occupancy_grid('C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map_updated.csv')
        #self.occupancy_grid = create_occupancy_grid(file_path, self.map_size[0], self.resolution)
        self.pathfinder = AStarPathfinder(self.occupancy_grid, self.resolution, self.map_size)
        self.data_file_path = 'robot_movement_data.csv'  # Define the path for the output CSV file
        self.data_file = None
        self.data_writer = None
        self.open_data_file()
        self.obstacle_positions = []
        self.obstacle_positions = load_obstacle_positions("C:/Users/abeda/webot-mapping-pathfinding/controllers/e_puck_controller/simulation_data.csv")

    def plan_and_follow_path(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return

        path = self.pathfinder.find_path(start_position, goal_position)
        #path = self.pathfinder.smooth_path(path)

        if path is None:
            print("No valid path found.")
            return

        path = simplify_path_by_angle(path, angle_threshold=170)  # Adjust threshold as needed
        self.visualize_path(path)

        waypoints = [self.pathfinder.grid_to_world(grid_position) for grid_position in path]
        self.move_robot_to_waypoints(waypoints)
    """
    def _move_towards_waypoint(self, current_position, target_position):
        

        while not reached_waypoint(current_position, target_position):
            self.update_pid(current_position, target_position)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()
            self.data_writer.writerow([self._robot.getTime(), current_position[0], current_position[1], current_position[2], target_position])
    """
    def reached_waypoint(self,curr_pos, target_pos, threshold=0.2):
        return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

    def _move_towards_waypoint(self, current_position, target_position):
        while not self.reached_waypoint(current_position, target_position):
            self.update_pid(current_position, target_position)
            
            # Solve the MPC problem to get the optimal control input
            control_input = self.solve_mpc(current_position, target_position, self.obstacle_positions)
            
            # Apply the control input to the robot
            self.set_robot_velocity(control_input[0], control_input[1])
            
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()
            
            self.data_writer.writerow([self._robot.getTime(), current_position[0], current_position[1], current_position[2], target_position])

    def set_robot_velocity(self, v, omega):
        # Set the velocities of the robot's motors based on v (linear velocity) and omega (angular velocity)
        left_speed = (v - omega * self.distance_between_wheels / 2.0) / self.wheel_radius
        right_speed = (v + omega * self.distance_between_wheels / 2.0) / self.wheel_radius
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)


    def visualize_path(self, path):
        grid = np.array(self.occupancy_grid)
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

    def check_for_future_collisions(self, current_position, target_position, obstacle_positions, velocity, steps_ahead=3):
        future_robot_positions = predict_future_positions(current_position, target_position, velocity, steps_ahead)
        current_time_step = int(self.robot.getTime() / self._timestep)
        future_obstacle_positions = get_future_obstacle_positions(obstacle_positions, current_time_step, steps_ahead)
        
        for future_robot_pos in future_robot_positions:
            for obstacle_pos in future_obstacle_positions:
                distance = np.linalg.norm(np.array(future_robot_pos[:2]) - np.array(obstacle_pos[:2]))
                if distance < SAFE_DISTANCE_THRESHOLD:
                    return True
        return False

    def kinematic_model(self, x, y, theta, v, omega, dt):
        # Simplified linear approximation:
        x_next = x + v * dt
        y_next = y + v * theta * dt
        theta_next = theta + omega * dt
        return cp.vstack([x_next, y_next, theta_next])


    def mpc_cost_function(self, predicted_states, control_inputs, goal_position, obstacles_positions):
        cost = 0
        for i in range(len(predicted_states)):
            state = predicted_states[i]
            control = control_inputs[i]

            # Distance to goal
            goal_cost = cp.norm(np.array(state[:2]) - np.array(goal_position[:2]))

            # Control effort
            control_cost = cp.norm(control)

            # Obstacle avoidance
            obstacle_cost = 0
            t_s = int((self.robot.getTime() * 100) / 64)
            print(t_s)
            obstacle_cost += 1 / cp.norm(np.array(state[:2]) - np.array(obstacles_positions[t_s][1][:2]))

            # Accumulate costs
            cost += goal_cost + control_cost + obstacle_cost

        return cost

    def solve_mpc(self, current_state, goal_position, obstacles_positions):
        N = 10  # Prediction horizon
        dt = self._timestep / 64  # Convert timestep to seconds

        # Variables for the state and control input over the horizon
        x = cp.Variable((N, 3))  # State variables [x, y, theta]
        u = cp.Variable((N, 2))  # Control variables [v, omega]

        # Cost function
        cost = 0
        constraints = []
        for t in range(N):
            if t == 0:
                cost += self.mpc_cost_function([current_state], u[t], goal_position, obstacles_positions)
            else:
                x_next = self.kinematic_model(x[t-1, 0], x[t-1, 1], x[t-1, 2], u[t-1, 0], u[t-1, 1], dt)

                constraints += [cp.reshape(x[t], (3, 1)) == x_next]  # Ensure shapes match
                cost += self.mpc_cost_function(x[t], u[t], goal_position, obstacles_positions)

            # Add control constraints here if necessary
            constraints += [cp.norm(u[t], 'inf') <= 1]  # Example constraint

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        return u.value[0, :]  # Return the first control input


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
            [pos for time_step, pos, _ in obstacle_positions if time_step == i]
        )
    return future_positions





