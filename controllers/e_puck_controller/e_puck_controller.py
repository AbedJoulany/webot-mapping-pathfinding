from controller import Robot, Supervisor
import math
import csv
import sys
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

        self.Kp_distance = 1.0
        self.Ki_distance = 0.01
        self.Kd_distance = 0.1

        self.Kp_heading = 0.1
        self.Ki_heading = 0.01  # Reduced the integral gain
        self.Kd_heading = 0.05

        self.distance_error_sum = 0
        self.previous_distance_error = 0

        self.heading_error_sum = 0
        self.previous_heading_error = 0

        self.data = []
        header = ['Time', 'Obstacle_X', 'Obstacle_Y', 'Obstacle_Theta']

        with open('simulation_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

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

    def step(self):
        pos = self.get_position()
        current_position = pos[:2]
        current_orientation = pos[2]

        self.collect_data(self.robot.getTime(), pos)

        if self.current_target_index < len(self.path):
            target = self.path[self.current_target_index]
            distance_error, heading_error = self.calculate_control_signal(target, current_position, current_orientation)

            #print(f"Target: {target}, Current Position: {current_position}, Distance Error: {distance_error}, Heading Error: {heading_error}")

            if distance_error < 0.01:
                #print("Reached target:", target)
                self.current_target_index += 1
                if self.current_target_index >= len(self.path):
                    print("Final target reached. Stopping.")
                    target_reached = True
                    for wheel in self.wheels:
                        wheel.setVelocity(0)
                    return

            # PID control for distance
            self.distance_error_sum += distance_error
            distance_error_delta = distance_error - self.previous_distance_error
            self.previous_distance_error = distance_error

            linear_velocity = (self.Kp_distance * distance_error + 
                               self.Ki_distance * self.distance_error_sum + 
                               self.Kd_distance * distance_error_delta)

            # PID control for heading
            self.heading_error_sum += heading_error
            heading_error_delta = heading_error - self.previous_heading_error
            self.previous_heading_error = heading_error

            angular_velocity = (self.Kp_heading * heading_error + 
                                self.Ki_heading * self.heading_error_sum + 
                                self.Kd_heading * heading_error_delta)

            #print(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")
            
            left_speed = self.limit_speed(linear_velocity - angular_velocity, -6.28, 6.28)
            right_speed = self.limit_speed(linear_velocity + angular_velocity, -6.28, 6.28)

            #print(f"Left Speed: {left_speed}, Right Speed: {right_speed}")
            
            self.wheels[0].setVelocity(left_speed)
            self.wheels[1].setVelocity(right_speed)
        else:
            for wheel in self.wheels:
                wheel.setVelocity(0)
                
    def run(self):
        while self.robot.step(self.time_step) != -1 and not target_reached:
            self.step()
        # Save collected data to CSV
        with open('simulation_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

target_reached = False
TIME_STEP = 64
robot_controller = EpuckController(TIME_STEP)
robot_controller.run()

"""if __name__ == "__main__":

        # Create instance of RobotController
        path_finding_robot_controller = PathFindingRobotController("E-PUCK-OBS")
        # Example start and goal positions
        start_position = path_finding_robot_controller.get_robot_pose_from_webots()
        x,y = path_finding_robot_controller.get_target_position()
        goal_position = (x,y,1.0)
        # Plan and follow path
        path_finding_robot_controller.plan_and_follow_path(start_position, goal_position)"""