import numpy as np
import heapq
import math

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
        x, y,z = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_traversable(self, position):
        x, y,z = position
        return self.occupancy_grid[y][x] == 0

    def reconstruct_path(self, current_node):
        print("at reconstruct_path")
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
                #return
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
                print(f"Adding neighbor: {neighbor_node.position} to open list with f-value: {neighbor_node.f}")

        return None  # No path found

    def world_to_grid(self, point):

        x, y,z = point

        #x_grid = -1*((x ) * self.resolution - self.map_size[0] / 2)
        #y_grid = ((y ) * self.resolution - self.map_size[0] / 2)

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
