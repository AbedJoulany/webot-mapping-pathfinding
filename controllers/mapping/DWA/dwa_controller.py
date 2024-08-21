import math
import numpy as np

class DWA:
    def __init__(self, config):
        self.max_speed = config['max_speed']
        self.min_speed = config['min_speed']
        self.max_yaw_rate = config['max_yaw_rate']
        self.max_accel = config['max_accel']
        self.max_delta_yaw_rate = config['max_delta_yaw_rate']
        self.velocity_resolution = config['velocity_resolution']
        self.yaw_rate_resolution = config['yaw_rate_resolution']
        self.predict_time = config['predict_time']
        self.to_goal_cost_gain = config['to_goal_cost_gain']
        self.obstacle_cost_gain = config['obstacle_cost_gain']
        self.speed_cost_gain = config['speed_cost_gain']
        self.dt = config['dt']

    def calculate_dynamic_window(self, current_velocity):
        # [min_speed, max_speed, min_yaw_rate, max_yaw_rate]
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]

        # Current speed dynamic window
        Vd = [
            current_velocity[3] - self.max_accel * self.dt,
            current_velocity[3] + self.max_accel * self.dt,
            current_velocity[4] - self.max_delta_yaw_rate * self.dt,
            current_velocity[4] + self.max_delta_yaw_rate * self.dt
        ]

        # Final dynamic window
        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3])
        ]

        return dw

    def motion(self, x, u):
        # Predict the new state
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def predict_trajectory(self, initial_state, v, w):
        trajectory = [initial_state]
        time = 0
        while time <= self.predict_time:
            initial_state = self.motion(initial_state, [v, w])
            trajectory.append(initial_state)
            time += self.velocity_resolution
        return trajectory

    def calculate_to_goal_cost(self, trajectory, goal):
        last_position = trajectory[-1]  # Get the last state in the trajectory
        dx = goal[0] - last_position[0]
        dy = goal[1] - last_position[1]
        goal_angle = math.atan2(dy, dx)
        heading_error = goal_angle - last_position[2]
        cost = abs(math.atan2(math.sin(heading_error), math.cos(heading_error)))
        return cost


    def calculate_obstacle_cost(self, trajectory, obstacles, threshold=0.5):
        min_distance = float("inf")
        for step in trajectory:
            for obs in obstacles:
                dist = np.linalg.norm(np.array([step[0], step[1]]) - np.array(obs))
                if dist < min_distance:
                    min_distance = dist
        if min_distance < threshold:
            return float("inf")  # Collision imminent
        return 1.0 / min_distance if min_distance != float("inf") else 0.0

    def calculate_speed_cost(self, v):
        return self.max_speed - v

    def dwa_control(self, current_state, goal, obstacles):
        dw = self.calculate_dynamic_window(current_state)
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = [current_state]

        for v in np.arange(dw[0], dw[1], self.velocity_resolution):
            for w in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                trajectory = self.predict_trajectory(current_state, v, w)

                to_goal_cost = self.calculate_to_goal_cost(trajectory, goal)
                obstacle_cost = self.calculate_obstacle_cost(trajectory, obstacles)
                speed_cost = self.calculate_speed_cost(v)

                final_cost = self.to_goal_cost_gain * to_goal_cost + \
                             self.obstacle_cost_gain * obstacle_cost + \
                             self.speed_cost_gain * speed_cost

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_trajectory = trajectory

        return best_u, best_trajectory

