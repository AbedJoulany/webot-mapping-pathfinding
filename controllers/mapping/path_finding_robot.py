import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt
from a_star import AStarPathfinder
from base_robot_controller import BaseRobotController

class PathFindingRobotController(BaseRobotController):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):  # Avoid reinitialization
            super().__init__()
            self._initialize_path_planning()
            self._initialized = True

    def _initialize_path_planning(self):
        # Accessing resolution and map_size from the base class

        # Usage example
        file_path = 'map.csv'

        self.occupancy_grid = create_occupancy_grid(file_path, self.map_size[0], self.resolution)
        #self.occupancy_grid = [[0] * int(self.map_size[0] / self.resolution) for _ in range(int(self.map_size[1] / self.resolution))]
        self.pathfinder = AStarPathfinder(self.occupancy_grid, self.resolution, self.map_size)

    def plan_and_follow_path(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return

        path = self.pathfinder.find_path(start_position, goal_position)
        if path is None:
            print("No valid path found.")
            return

        waypoints = [self.pathfinder.grid_to_world(grid_position) for grid_position in path]
        self.move_robot_to_waypoints(waypoints)



