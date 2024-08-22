import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt
from rrt_star import RRTStar, Node
from base_robot_controller import BaseRobotController

class PathFindingRobotControllerRRT(BaseRobotController):
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
        file_path = 'map.csv'
        self.occupancy_grid = create_occupancy_grid(file_path, self.map_size[0], self.resolution)
        self.pathfinder = RRTStar(
            start=(0, 0),
            goal=(0, 0),
            obstacle_list=self.occupancy_grid,
            map_size=self.map_size,
            resolution=self.resolution,
            step_size=0.1,
            max_iterations=500,
            expand_dis=0.2
        )

    def plan_and_follow_path(self, start_position, goal_position):
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print("Error: Start or goal position contains NaN values.")
            return

        # Update the start and goal positions for RRT*
        self.pathfinder.start = Node(start_position)
        self.pathfinder.goal = Node(goal_position)
        self.pathfinder.node_list = [self.pathfinder.start]  # Reset node list

        path = self.pathfinder.planning()
        if path is None:
            print("No valid path found.")
            return

        waypoints = [self.pathfinder.grid_to_world(grid_position) for grid_position in path]
        self.move_robot_to_waypoints(waypoints)



