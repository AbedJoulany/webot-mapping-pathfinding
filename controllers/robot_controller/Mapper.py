import math
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from controller import Robot


class Mapper:
    def __init__(self, grid_size, grid_resolution, max_sensor_range):
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.max_sensor_range = max_sensor_range
        self.map_size = grid_size * grid_resolution
        self.grid = np.zeros((grid_size, grid_size))

    def update_map(self, x, y, sensor_angle, sensor_distance):
        grid_x = int((x + self.map_size / 2) / self.grid_resolution)
        grid_y = int((y + self.map_size / 2) / self.grid_resolution)

        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            obs_x = x + sensor_distance * math.cos(sensor_angle)
            obs_y = y + sensor_distance * math.sin(sensor_angle)

            grid_obs_x = int((obs_x + self.map_size / 2) / self.grid_resolution)
            grid_obs_y = int((obs_y + self.map_size / 2) / self.grid_resolution)

            if 0 <= grid_obs_x < self.grid_size and 0 <= grid_obs_y < self.grid_size:
                self.grid[grid_obs_x, grid_obs_y] = 1  # mark obstacle

    def find_frontiers(self):
        known = self.grid > 0
        unknown = self.grid == 0
        dilated_known = binary_dilation(known, iterations=1)
        frontiers = np.logical_and(dilated_known, unknown)
        return np.argwhere(frontiers)
    
    def display_map(self):
        plt.imshow(self.grid, cmap='gray')
        plt.show()

