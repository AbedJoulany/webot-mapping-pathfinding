import math
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from controller import Robot
from Sensor import Sensor

from Localization import Localization
from Motor import Motor

from Mapper import Mapper

class RobotController:
    def __init__(self):
        self.robot = Robot()
        self.localization = Localization(self.robot)
        self.mapper = Mapper(GRID_SIZE, GRID_RESOLUTION, MAX_SENSOR_RANGE)

        self.motors = []
        motor_names = ['bk_right_wheel', 'fr_right_wheel', 'bk_left_wheel', 'fr_left_wheel']
        for name in motor_names:
            self.motors.append(Motor(self.robot, name))

    def set_motor_speeds(self, left_speed, right_speed):
        for i in range(2):
            self.motors[i].set_velocity(right_speed)
        for i in range(2, 4):
            self.motors[i].set_velocity(left_speed)

    def move_towards_goal(self, goal_x, goal_y):
        gps_position = self.localization.get_position()
        current_x, current_y, _ = gps_position
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)

        yaw = self.localization.get_orientation()[2]
        turn_angle = angle_to_goal - yaw

        while turn_angle > math.pi:
            turn_angle -= 2 * math.pi
        while turn_angle < -math.pi:
            turn_angle += 2 * math.pi

        if abs(turn_angle) > 0.1:
            if turn_angle > 0:
                self.set_motor_speeds(-1.0, 1.0)
            else:
                self.set_motor_speeds(1.0, -1.0)
        else:
            self.set_motor_speeds(1.0, 1.0)

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            gps_position = self.localization.get_position()
            imu_orientation = self.localization.get_orientation()
            ds_left_value = self.localization.get_left_distance()
            ds_right_value = self.localization.get_right_distance()

            print(f"GPS Position: {gps_position}")
            print(f"IMU Orientation: {imu_orientation}")
            print(f"Left Distance Sensor: {ds_left_value}")
            print(f"Right Distance Sensor: {ds_right_value}")

            x, y, _ = gps_position
            yaw = imu_orientation[2]

            if ds_left_value < MAX_SENSOR_RANGE:
                self.mapper.update_map(x, y, yaw + math.pi / 2, ds_left_value)
            if ds_right_value < MAX_SENSOR_RANGE:
                self.mapper.update_map(x, y, yaw - math.pi / 2, ds_right_value)

            frontiers = self.mapper.find_frontiers()
            if len(frontiers) > 0:
                goal = frontiers[0]
                goal_pos = (goal[0] * GRID_RESOLUTION - self.mapper.map_size / 2,
                            goal[1] * GRID_RESOLUTION - self.mapper.map_size / 2)
                print(f"Moving towards goal: {goal_pos}")
                self.move_towards_goal(goal_pos[0], goal_pos[1])
            else:
                self.set_motor_speeds(0.0, 0.0)

        self.mapper.display_map()


if __name__ == "__main__":
    TIME_STEP = 64
    GRID_SIZE = 100
    GRID_RESOLUTION = 0.1
    MAX_SENSOR_RANGE = 0.2

    controller = RobotController()
    controller.run()