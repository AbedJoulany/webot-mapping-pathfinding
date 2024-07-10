import heapq
import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position[0] == other.position[0] and self.position[1] == other.position[1]


    def __lt__(self, other):
        return self.f < other.f

# Define A* Pathfinding class
class AStarPathfinder:
    def __init__(self, occupancy_grid, resolution, map_size):
        self.occupancy_grid = occupancy_grid
        self.resolution = resolution
        self.map_size = map_size
        self.grid_size = map_size[0] / resolution
        print(f"len(self.grid_size) {self.grid_size}")
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
    def is_within_bounds(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_traversable(self, position):
        x, y = position
        return self.occupancy_grid[y][x] == 0

    def reconstruct_path(self, current_node):
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]  # Return reversed path

    def find_path(self, start, goal):
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        start_node = Node(None, start_grid)
        goal_node = Node(None, goal_grid)

        print(f"start_node {start_node.position}")
        print(f"goal_node {goal_node.position}")


        open_list = []
        closed_list = set()

        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)
            print(f"Current node: {current_node.position}")
            closed_list.add(current_node.position)

            if current_node == goal_node:
                print("Goal reached!")
                return self.reconstruct_path(current_node)

            for action in self.actions:
                neighbor_pos = (current_node.position[0] + action[0], current_node.position[1] + action[1])
                if not self.is_within_bounds(neighbor_pos) or not self.is_traversable(neighbor_pos):
                    continue

                neighbor_node = Node(current_node, neighbor_pos)
                if neighbor_node.position in closed_list:
                    continue

                # More accurate cost calculation
                if action[0] == 0 or action[1] == 0:
                    neighbor_node.g = current_node.g + 1  # Straight movement cost
                else:
                    neighbor_node.g = current_node.g + math.sqrt(2)  # Diagonal movement cost

                neighbor_node.h = self.heuristic(neighbor_node.position, goal_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                if not any(open_node for open_node in open_list if neighbor_node == open_node and neighbor_node.g > open_node.g):
                    heapq.heappush(open_list, neighbor_node)
                
                print(f"Adding neighbor: {neighbor_node.position} to open list with f-value: {neighbor_node.f}")

        return None  # No path found

    def world_to_grid(self, point):

        x, y,z = point
        x_grid = int((x + self.map_size[0] / 2) / self.resolution)
        y_grid = int((y + self.map_size[0] / 2 ) / self.resolution)

        return (x_grid, y_grid,z)

    def grid_to_world(self, point):
        x_world = point[0] * self.resolution - self.map_size[0] / 2
        y_world = point[1] * self.resolution - self.map_size[1] / 2
        z_world = point[2]
        return (x_world, y_world, z_world)

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)    
        #return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
# RobotController class with A* integration
class RobotController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._robot = Supervisor()
            cls._instance._timestep = int(cls._instance._robot.getBasicTimeStep())
            cls._instance._initialize_devices()
            cls._instance._initialize_path_planning()  # Initialize path planning
        return cls._instance

    def _initialize_devices(self):
        # e-puck
        self._epuckRobot = self._robot.getFromDef("e-puck")
        self._rotation_field = self._epuckRobot.getField("rotation")
        
        self._lidar = self._robot.getDevice("lidar")
        self._lidar.enable(self._timestep)
        self._lidar.enablePointCloud()
        
        self._gps = self._robot.getDevice("gps")
        self._gps.enable(self._timestep)
        
        # Compass
        self._compass = self._robot.getDevice("compass")
        self._compass.enable(self._timestep)

        self._left_motor = self._robot.getDevice("left wheel motor")
        self._right_motor = self._robot.getDevice("right wheel motor")
        self._left_motor.setPosition(float('inf'))
        self._right_motor.setPosition(float('inf'))
        
        self.left_ps = self._robot.getDevice("right wheel sensor")
        self.left_ps.enable(self._timestep)

        self.right_ps = self._robot.getDevice("left wheel sensor")
        self.right_ps.enable(self._timestep)
        
        self._keyboard = self._robot.getKeyboard()
        self._keyboard.enable(self._timestep)

        self.ps_values = [0, 0]
        self.dist_values = [0, 0]

        self.wheel_radius = 0.0205
        self.distance_between_wheels = 0.052
        self.wheel_circumference = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circumference / (2 * 3.14)

        #self.robot_pose_encoder = self.get_robot_pose_from_webots()
        self.last_ps_values = [0, 0]

        self.keys = {key: False for key in ["w", "a", "s", "d", "o", "m", "h", "t", "y"]}
        self.manual_control = {"active": True, "count": 0}
        self.path = {"path": 0}

        self.hTruePos = np.zeros((3, 0))
        self.hEncoPos = np.zeros((3, 0))
                
        self.map = np.zeros((0, 2))  # Initialize the map to store lidar points

        self.hedef = self._robot.getFromDef("target")  # Adding target initialization

        self.w = 0
        self.v = 0
        
        self.world_size = (2,2)
        self.map_size = (2,2)    

        if self._lidar is None:
            raise ValueError("Lidar device not found. Ensure the lidar device is correctly named and exists in the Webots world file.")

    def _initialize_path_planning(self):
        # Load occupancy grid
        #self.occupancy_grid = create_occupancy_grid(file_path, map_size, resolution)

        self.occupancy_grid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
]
        # Initialize A* pathfinder
        self.pathfinder = AStarPathfinder(self.occupancy_grid, resolution, (map_size,map_size))

    @property
    def robot(self):
        return self._robot

    @property
    def timestep(self):
        return self._timestep

    @property
    def lidar(self):
        return self._lidar

    @property
    def left_motor(self):
        return self._left_motor

    @property
    def right_motor(self):
        return self._right_motor
        
    @property
    def keyboard(self):
        return self._keyboard
    
    @property
    def gps(self):
        return self._gps
        
    @property
    def compass(self):
        return self._compass

    def set_motor_speeds(self, left_speed, right_speed):
        self._left_motor.setVelocity(left_speed)
        self._right_motor.setVelocity(right_speed)

    def get_gps_position(self):
        return self._gps.getValues()

    def get_compass_heading(self):
        compass_values = self._compass.getValues()
        heading = (compass_values[0], compass_values[2])
        return heading
    
    def get_robot_pose_from_webots(self):
        robot_pos = self._epuckRobot.getPosition()
        robot_rot = self._rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)

        return [
            round(robot_pos[0], 3),
            round(robot_pos[1], 3),
            robot_rot_z,
        ]

    def move_robot_to_waypoints(self, waypoints):
        current_position = self.get_robot_pose_from_webots()
        for waypoint in waypoints:
            print(f"Moving towards waypoint: {waypoint}")
            self._move_towards_waypoint(current_position, waypoint)
            current_position = waypoint
        # Stop the robot after reaching the last waypoint
        self.set_motor_speeds(0, 0)

    def get_target_position(self):
        hedef_pos = self.hedef.getPosition()
        return np.array([
            round(hedef_pos[0], 3),
            round(hedef_pos[1], 3)
        ])
    
    def follow_path(self, path):
        current_position = self.get_robot_pose_from_webots()
        for grid_position in path:
            # Convert grid position to world coordinates
            target_position = self.pathfinder.grid_to_world(grid_position)
            print(f"Following path to grid position: {grid_position}, target_position: {target_position}")
            # Move towards the next waypoint
            self._move_towards_waypoint(current_position, target_position)
            current_position = target_position

        # Stop the robot after reaching the last waypoint
        self.set_motor_speeds(0, 0)

    def _move_towards_waypoint(self, current_position, target_position):
        current_heading = current_position[2]
        print(f"current_position {current_position}")
        print(f"current_heading {current_heading}")
        target_heading = math.atan2(target_position[1] - current_position[1], target_position[0] - current_position[0])

        while abs(target_heading - current_heading) > 0.01:
            # Adjust heading towards the target
            print("Adjusting heading...")
            self.set_motor_speeds(1.0, -1.0)
            current_heading = self.get_robot_pose_from_webots()[2]
            print(f"Current heading updated: {current_heading}")
        # Move forward until reaching the target position
        while np.linalg.norm(current_position[:2] - target_position[:2]) > 0.05:            # Move forward with adjusted speeds
            # Move forward with adjusted speeds
            print("Moving forward...")
            self.set_motor_speeds(6.28, 6.28)
            current_position = self.get_robot_pose_from_webots()
            print(f"Current position updated: {current_position}")

    def plan_and_follow_path(self, start_position, goal_position):

        # Check if start_position or goal_position contains NaN
        if np.any(np.isnan(start_position)) or np.any(np.isnan(goal_position)):
            print(f"start_position = {start_position}")
            print(f"goal_position = {goal_position}")
            print("Error: Start or goal position contains NaN values.")
            return

        # Proceed with path planning and following
        path = self.pathfinder.find_path(start_position, goal_position)
        if path is not None:
            self.follow_path(path)
        else:
            print("No valid path found.")
            return

        # Convert path to GPS waypoints
        waypoints = []
        for grid_position in path:
            point = self.pathfinder.grid_to_world(grid_position)
            print(f"point = {point}")
            waypoints.append(point)  # Assuming y-coordinate is 0 (flat ground)

        # Move robot along the planned path
        self.move_robot_to_waypoints(waypoints)

# Example usage
file_path = 'map.csv'
map_size = 2  # meters
resolution = 0.5  # meter per cell


