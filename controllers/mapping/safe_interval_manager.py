class SafeIntervalManager:
    def __init__(self, map_size, resolution):
        self.safe_intervals = {}  # Dictionary to store safe intervals for each cell
        self.grid_size = map_size[0] / resolution
        self.resolution = resolution
    def update_safe_intervals(self, obstacle_positions, time_horizon, current_timestep):
        """
        Updates safe intervals based on the predicted positions of dynamic obstacles.
        """
        # Example of how you might update intervals for each obstacle
        for time in range(time_horizon):
            time = time * 0.032
            future_position = self.get_future_obstacle_position(obstacle_positions, time + current_timestep)
            grid_position = self.world_to_grid(future_position)
            if grid_position not in self.safe_intervals:
                self.safe_intervals[grid_position] = []
            # Mark this interval as unsafe (adjust time ranges as necessary)
            self.safe_intervals[grid_position].append((time, time + 0.032))

    def world_to_grid(self, point):
        x, y,z = point
        x_grid = int((x + self.grid_size / 2) / self.resolution)
        y_grid = int((y + self.grid_size / 2 ) / self.resolution)

        return (x_grid, y_grid,z)

    def get_safe_intervals(self, position):
        """
        Retrieves the safe intervals for a given position.
        """
        return self.safe_intervals.get(position, [(0, float('inf'))])

    def predict_future_position(self, obstacle = (0,0,0,0), time = 0):
        """
        Predicts the future position of the obstacle after a given time interval.

        :param obstacle: A tuple containing the current position (x, y) and velocity (vx, vy) of the obstacle.
        :param time: The time interval for which the future position should be predicted.
        :return: A tuple representing the predicted future position (x, y) of the obstacle.
        """
        x, y, vx, vy = obstacle
        future_x = x + vx * time
        future_y = y + vy * time
        return (future_x, future_y, 0)


    def get_future_obstacle_position(self, obstacle_positions, current_time_step):
        current_time_step = int(current_time_step *1000 /64)-1
        return obstacle_positions[current_time_step][1]  # Handle case where no obstacles found


    def is_path_obstructed(self, path, time_horizon,current_time_step, obstacle_positions):
        """
        Checks if the path is obstructed by any obstacles within a given time horizon.

        :param path: A list of tuples representing the planned path [(x1, y1, z1), (x2, y2, z2), ...].
        :param time_horizon: The time horizon to predict the obstacles' future positions.
        :param obstacle_positions:
        :return: True if the path is obstructed, False otherwise.
        """
        for t in range(1, time_horizon + 1):
            time = time * 0.064
            future_position = self.get_future_obstacle_position(obstacle_positions, current_time_step + t)
            grid_position = self.world_to_grid(future_position)

            for path_point in path:
                path_grid_position = self.world_to_grid(path_point)

                if grid_position[:2] == path_grid_position[:2]:  # Compare only x and y grid positions
                    safe_intervals = self.get_safe_intervals(path_grid_position)
                    buffer_distance = 0.1  # Adjust this value as needed for your environment

                    if (abs(grid_position[0] - path_grid_position[0]) <= buffer_distance and
                        abs(grid_position[1] - path_grid_position[1]) <= buffer_distance and
                        not any(start <= t <= end for start, end in safe_intervals)):
                        # If the future position of the obstacle intersects with the path within a buffer zone
                        # and it's within an unsafe time interval, the path is obstructed.
                        return True

        return False  # No obstruction detected within the time horizon
