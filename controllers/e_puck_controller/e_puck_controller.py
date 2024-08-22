from controller import Robot, Supervisor
import math
import csv
import sys

from pid_controller import PIDController
#sys.path.append('C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping')
#from mapping import PathFindingRobotController

class EpuckController:
    def __init__(self, time_step):
        self.robot = Supervisor()
        self.time_step = time_step
        self.epuck_robot = self.robot.getFromDef("E-PUCK-OBS")
        self.rotation_field = self.epuck_robot.getField("rotation")
        self.wheels = []
        wheelsNames = ['left wheel motor', 'right wheel motor']
        for i in range(2):
            wheel = self.robot.getDevice(wheelsNames[i])
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)

        self.path = [(-0.25, -0.25), (-0.25, 0.75), (0.9, 0.9), (0.75, -0.25)]
        self.current_target_index = 0

        self.distance_pid = PIDController(2.0, 0.02, 0.2)
        self.heading_pid = PIDController(0.1, 0.01, 0.05)

        self.distance_error_sum = 0
        self.previous_distance_error = 0

        self.heading_error_sum = 0
        self.previous_heading_error = 0

        self.data = []
        header = ['Time', 'Obstacle_X', 'Obstacle_Y', 'Obstacle_Theta']

        """
        with open('simulation_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        """

    def get_position(self):
        robot_pos = self.epuck_robot.getPosition()
        robot_rot = self.rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)
        return [round(robot_pos[0], 3), round(robot_pos[1], 3), robot_rot_z]

    def get_velocity(self):
        velocity = self.epuck_robot.getVelocity()
        return [round(velocity[0], 3), round(velocity[1], 3), round(velocity[2], 3)]

    def limit_speed(self, speed, min_speed, max_speed):
        return max(min_speed, min(speed, max_speed))

    def calculate_error(self, target, current):
        return math.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

    def calculate_control_signal(self, target, current_position, current_orientation):
        angle_to_target = math.degrees(math.atan2(target[1] - current_position[1], target[0] - current_position[0]))
        heading_error = angle_to_target - math.degrees(current_orientation)

        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        distance_error = self.calculate_error(target, current_position)

        return distance_error, heading_error

    def collect_data(self, time_step, pos):
        self.data.append([
            time_step,
            round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)
        ])


    def update_pid(self, current_position, target_position):
        current_orientation = current_position[2]

        distance_error, heading_error = self.calculate_control_signal(target_position, current_position, current_orientation)

        dt = self.time_step / 64
        linear_velocity = self.distance_pid.update(distance_error, dt)
        angular_velocity = self.heading_pid.update(heading_error, dt)

        left_speed = self.limit_speed(linear_velocity - angular_velocity, -6.28, 6.28)
        right_speed = self.limit_speed(linear_velocity + angular_velocity, -6.28, 6.28)

        self.wheels[0].setVelocity(left_speed)
        self.wheels[1].setVelocity(right_speed)

    def calculate_error(self, target, current):
        return math.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

    def calculate_control_signal(self, target, current_position, current_orientation):
        angle_to_target = math.degrees(math.atan2(target[1] - current_position[1], target[0] - current_position[0]))
        heading_error = angle_to_target - math.degrees(current_orientation)

        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        distance_error = self.calculate_error(target, current_position)

        return distance_error, heading_error

    def step(self):
        pos = self.get_position()
        current_position = pos[:2]
        current_orientation = pos[2]

        self.collect_data(self.robot.getTime(), pos)

        if self.current_target_index < len(self.path):
            target = self.path[self.current_target_index]
            distance_error, heading_error = self.calculate_control_signal(target, current_position, current_orientation)

            if distance_error < 0.01:
                #print("Reached target:", target)
                self.current_target_index += 1
                if self.current_target_index >= len(self.path):
                    print("Final target reached. Stopping.")
                    target_reached = True
                    for wheel in self.wheels:
                        wheel.setVelocity(0)
                    return

            self.update_pid(pos, target)
        else:
            for wheel in self.wheels:
                wheel.setVelocity(0)

    def run(self):
        while self.robot.step(self.time_step) != -1 and not target_reached:
            self.step()
        # Save collected data to CSV
        """
        with open('simulation_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)
        """

target_reached = False

TIME_STEP = 64
robot_controller = EpuckController(TIME_STEP)
robot_controller.run()
