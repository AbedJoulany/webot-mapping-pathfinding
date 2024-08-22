import random
import numpy as np
import math

class Node:
    def __init__(self, position):
        self.position = position
        self.cost = 0.0
        self.parent = None

class RRTStar:
    def __init__(self, start, goal, obstacle_list, map_size, resolution=0.01, step_size=0.1, max_iterations=500, expand_dis=0.2):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list
        self.map_size = map_size
        self.resolution = resolution
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.node_list = [self.start]
        self.expand_dis = expand_dis
    
    def planning(self):
        for _ in range(self.max_iterations):
            sampled_point = self.sample_free()
            nearest_node = self.get_nearest_node(sampled_point)
            new_node = self.steer(nearest_node, sampled_point, self.expand_dis)

            if not self.collision_check(nearest_node, new_node):
                continue

            near_nodes = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_nodes)
            self.node_list.append(new_node)
            self.rewire(new_node, near_nodes)

            if np.linalg.norm(np.array(new_node.position) - np.array(self.goal.position)) < self.step_size:
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(self.goal.position))
                self.node_list.append(self.goal)
                return self.generate_final_path()

        return None  # No path found

    def steer(self, from_node, to_point, extend_length=float("inf")):
        direction = np.array(to_point) - np.array(from_node.position)
        distance = np.linalg.norm(direction)
        direction = direction / distance

        new_position = tuple(np.array(from_node.position) + min(extend_length, distance) * direction)
        new_node = Node(new_position)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(np.array(new_position) - np.array(from_node.position))
        
        return new_node

    def sample_free(self):
        return (random.uniform(-self.map_size[0] / 2, self.map_size[0] / 2), random.uniform(-self.map_size[1] / 2, self.map_size[1] / 2), 0)

    def get_nearest_node(self, sampled_point):
        return min(self.node_list, key=lambda node: np.linalg.norm(np.array(node.position) - np.array(sampled_point)))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.position[0] - from_node.position[0]
        dy = to_node.position[1] - from_node.position[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    def collision_check(self, nearest_node, new_node):
        x1, y1 = self.world_to_grid(nearest_node.position)
        x2, y2 = self.world_to_grid(new_node.position)
        
        # Bresenham's line algorithm to discretize the path
        points = self.bresenham(x1, y1, x2, y2)

        for x, y in points:
            if self.obstacle_list[y, x] == 1:
                return False
        return True

    def world_to_grid(self, position):
        x_grid = int((position[0] + self.map_size[0] / 2) / self.resolution)
        y_grid = int((position[1] + self.map_size[1] / 2) / self.resolution)
        return x_grid, y_grid
    
    def grid_to_world(self, position):
        x_world = position[0] * self.resolution - self.map_size[0] / 2
        y_world = position[1] * self.resolution - self.map_size[1] / 2
        z_world = position[2]
        return (x_world, y_world, z_world)
    
    def bresenham(self, x1, y1, x2, y2):
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points
    
    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        radius = self.expand_dis * np.sqrt((np.log(n) / n))
        near_nodes = [node for node in self.node_list if np.linalg.norm(np.array(node.position) - np.array(new_node.position)) <= radius]
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return new_node

        min_cost = new_node.cost
        best_node = new_node.parent
        for node in near_nodes:
            if self.collision_check(node, new_node):
                cost = node.cost + np.linalg.norm(np.array(node.position) - np.array(new_node.position))
                if cost < min_cost:
                    min_cost = cost
                    best_node = node
        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if self.collision_check(new_node, node):
                cost = new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(node.position))
                if cost < node.cost:
                    node.parent = new_node
                    node.cost = cost

    def generate_final_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.position)
            node = node.parent
        return path[::-1]
