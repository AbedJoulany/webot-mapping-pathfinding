import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController
from visualize_grid import create_occupancy_grid, load_occupancy_grid
import csv
import os

class PathFindingRobotController(BaseRobotController):
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

    def _move_towards_waypoint(self, current_position, target_position):
        def reached_waypoint(curr_pos, target_pos, threshold=0.2):
            return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

        while not reached_waypoint(current_position, target_position):
            self.update_pid(current_position, target_position)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()
            self.data_writer.writerow([self._robot.getTime(), current_position[0], current_position[1], current_position[2], target_position])



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