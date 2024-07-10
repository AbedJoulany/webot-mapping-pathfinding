import numpy as np
import heapq

from visualize_grid import create_occupancy_grid

class AStarPathfinder:
    def __init__(self, occupancy_grid, resolution, map_size):
        self.occupancy_grid = occupancy_grid
        self.resolution = resolution
        self.map_size = map_size
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def is_within_bounds(self, position):
        x, y = position
        return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]

    def is_traversable(self, position):
        x, y = position
        grid_x = int((x + self.map_size[0] / 2) / self.resolution)
        grid_y = int((y + self.map_size[1] / 2) / self.resolution)
        return self.occupancy_grid[grid_y, grid_x] == 0

    def heuristic_cost(self, current, goal):
        return np.linalg.norm(np.array(current) - np.array(goal))

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    def find_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current_cost, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, goal)

            for action in self.actions:
                neighbor = (current[0] + action[0], current[1] + action[1])

                if not self.is_within_bounds(neighbor):
                    continue

                if not self.is_traversable(neighbor):
                    continue

                tentative_g_score = g_score[current] + self.heuristic_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score, neighbor))

        return None  # No path found

# Example usage
file_path = 'map.csv'
map_size = 2  # meters
resolution = 0.001  # meter per cell

# Load occupancy grid
occupancy_grid = create_occupancy_grid(file_path, map_size, resolution)

# Initialize A* pathfinder
pathfinder = AStarPathfinder(occupancy_grid, resolution, (int(map_size / resolution), int(map_size / resolution)))

# Example start and goal positions
start_position = (0.0, 0.0)
goal_position = (1.0, 1.0)

# Find path
path = pathfinder.find_path(start_position, goal_position)
print("Found path:", path)
