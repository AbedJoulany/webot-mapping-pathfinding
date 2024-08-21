import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from safe_interval_manager import SafeIntervalManager

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None,time =0):
        self.parent = parent
        self.position = position
        self.time = time  # Time of arrival at this node


        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position[0] == other.position[0] and self.position[1] == other.position[1]

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash((self.position[0], self.position[1]))

# Define A* Pathfinding class
class AStarPathfinder:
    def __init__(self, occupancy_grid, resolution, map_size, robot_radius):
        self.occupancy_grid = np.copy(occupancy_grid)
        self.resolution = resolution
        self.map_size = map_size
        self.grid_size = map_size[0] / resolution
        self.robot_radius = 0.025 / resolution  # Convert robot radius to grid units


        #print(f"len(self.grid_size) {self.grid_size}")
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)
                        ,(-1, -1), (-1, 1), (1, -1), (1, 1)]


    def set_occupancy_grid(self, occupancy_grid):
        self.occupancy_grid = np.copy(occupancy_grid)

    def is_within_bounds(self, position):
        x, y,z = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    def is_traversable(self, position):
        x, y, z = position
        radius = int(self.robot_radius)

        # Calculate the corner positions
        corners = [
            (x - radius, y - radius),
            (x - radius, y + radius),
            (x + radius, y - radius),
            (x + radius, y + radius)
        ]

        # Check each corner
        for corner_x, corner_y in corners:
            if not self.is_within_bounds((corner_x, corner_y, z)) or self.occupancy_grid[corner_y][corner_x] != 0:
                return False
        return True


    def reconstruct_path(self, current_node):
        #print("at reconstruct_path")
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

        open_list = []
        closed_list = {}

        heapq.heappush(open_list, start_node)
        #plt.ion()
        while open_list:
            current_node = heapq.heappop(open_list)
            #print(f"Current node: {current_node.position}")
            #closed_list.add(current_node.position)
            closed_list[current_node.position] = current_node
            #self.visualize_pathfinding(open_list, closed_list, current_node, start, goal)
            if current_node == goal_node:
                #print("Goal reached!")
                #plt.ioff()
                return self.reconstruct_path(current_node)

            for action in self.actions:
                neighbor_pos = (current_node.position[0] + action[0], current_node.position[1] + action[1], 0)
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
                #print(f"Adding neighbor: {neighbor_node.position} to open list with f-value: {neighbor_node.f}")

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
        #return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def update_occupancy_grid_with_obstacles(self, obstacle_positions, obstacle_radius):
        """
        Updates the occupancy grid by clearing previous obstacle positions and marking new obstacle positions.

        :param obstacle_positions: List of tuples containing obstacle positions [(x1, y1), (x2, y2), ...]
        :param obstacle_radius: Radius of the obstacles to consider when marking the grid.
        """

        # Convert obstacle radius to grid cells
        grid_radius = int(obstacle_radius / self.resolution)

        # Mark the new obstacle positions
        for obs_pos in obstacle_positions:
            grid_x, grid_y,z = self.world_to_grid(obs_pos)

            # Mark all cells within the obstacle's radius as occupied
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    # Calculate the distance from the obstacle center
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    if distance <= grid_radius:
                        # Mark the cell as occupied if within radius
                        if 0 <= grid_x + dx < len(self.occupancy_grid[0]) and 0 <= grid_y + dy < len(self.occupancy_grid):
                            self.occupancy_grid[grid_y + dy][grid_x + dx] = 1  # Mark as occupied (1 for obstacle)


    def plot_occupancy_grid(self):
        """Plots the current state of the occupancy grid."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.occupancy_grid, cmap='Greys', origin='lower')
        plt.title("Occupancy Grid with Obstacles")
        plt.xlabel("X grid cells")
        plt.ylabel("Y grid cells")
        plt.colorbar(label='Occupancy')
        plt.show()


    def visualize_pathfinding(self, open_list, closed_list, current_node, start, goal):
            # Clear the current plot
            # Clear the current plot only if necessary
            if not hasattr(self, '_ax') or self._ax is None:
                self._fig, self._ax = plt.subplots()

            self._ax.clear()

            # Plot the occupancy grid
            self._ax.imshow(self.occupancy_grid, cmap='gray', origin='lower')

            # Plot the start and goal positions
            start_grid = self.world_to_grid(start)
            goal_grid = self.world_to_grid(goal)
            self._ax.plot(start_grid[0], start_grid[1], "go")  # Start is green
            self._ax.plot(goal_grid[0], goal_grid[1], "ro")    # Goal is red

            closed_positions = np.array([pos for pos, _ in closed_list.items()])
            open_positions = np.array([node.position for node in open_list])
            if len(closed_positions) > 0:
                self._ax.plot(closed_positions[:, 0], closed_positions[:, 1], "bx")  # Closed nodes are blue

            if len(open_positions) > 0:
                self._ax.plot(open_positions[:, 0], open_positions[:, 1], "yo")  # Open nodes are yellow

            # Highlight the current node
            self._ax.plot(current_node.position[0], current_node.position[1], "co")  # Current node is cyan

            # Draw the plot
            plt.draw()
            plt.pause(0.00001)  # Pause to allow the plot to update

