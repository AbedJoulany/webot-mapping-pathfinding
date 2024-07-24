from controller import Robot, Supervisor
import math
import csv

TIME_STEP = 64
robot = Supervisor()
four_wheels_robot = robot.getFromDef("FourWheelsRobot")
rotation_field = four_wheels_robot.getField("rotation")

def get_position():
    robot_pos = four_wheels_robot.getPosition()
    robot_rot = rotation_field.getSFRotation()
    axis_z = robot_rot[2]
    robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)
    return [round(robot_pos[0], 3), round(robot_pos[1], 3), robot_rot_z]

def limit_speed(speed, min_speed, max_speed):
    return max(min_speed, min(speed, max_speed))

wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for i in range(4):
    wheels.append(robot.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)

# Path of points to follow (in meters)
path = [(0, 0), (-1, -1), (0, 0), (1, 1), (0, 0)]
current_target_index = 0

# PID controller parameters
Kp_distance = 1.0
Ki_distance = 0.01
Kd_distance = 0.1

Kp_heading = 0.1
Ki_heading = 0.01
Kd_heading = 0.05

distance_error_sum = 0
previous_distance_error = 0

heading_error_sum = 0
previous_heading_error = 0

def calculate_error(target, current):
    return math.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

def calculate_control_signal(target, current_position, current_orientation):
    angle_to_target = math.degrees(math.atan2(target[1] - current_position[1], target[0] - current_position[0]))
    heading_error = angle_to_target - math.degrees(current_orientation)

    if heading_error > 180:
        heading_error -= 360
    elif heading_error < -180:
        heading_error += 360

    distance_error = calculate_error(target, current_position)

    return distance_error, heading_error

# Data collection setup
data = []
header = ['Time', 'Obstacle_X', 'Obstacle_Y', 'Obstacle_Theta']

with open('simulation_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

def collect_data(time_step, pos):
    data.append([
        time_step,
        round(pos[0], 3),round(pos[1], 3), round(pos[2], 3)
    ])

while robot.step(TIME_STEP) != -1:
    pos = get_position()
    current_position = pos[:2]
    current_orientation = pos[2]

    collect_data(robot.getTime(), pos)

    if current_target_index < len(path):
        target = path[current_target_index]
        distance_error, heading_error = calculate_control_signal(target, current_position, current_orientation)

        if distance_error < 0.1:
            print("reached")
            current_target_index += 1
            if current_target_index >= len(path):
                print("in break")
                wheels[0].setVelocity(0)
                wheels[1].setVelocity(0)
                wheels[2].setVelocity(0)
                wheels[3].setVelocity(0)
                break
            continue

        # PID control for distance
        distance_error_sum += distance_error
        distance_error_delta = distance_error - previous_distance_error
        previous_distance_error = distance_error

        linear_velocity = (Kp_distance * distance_error + 
                           Ki_distance * distance_error_sum + 
                           Kd_distance * distance_error_delta)

        # PID control for heading
        heading_error_sum += heading_error
        heading_error_delta = heading_error - previous_heading_error
        previous_heading_error = heading_error

        angular_velocity = (Kp_heading * heading_error + 
                            Ki_heading * heading_error_sum + 
                            Kd_heading * heading_error_delta)
        
        left_speed = limit_speed(linear_velocity - angular_velocity, -10, 10)
        right_speed = limit_speed(linear_velocity + angular_velocity, -10, 10)
        
        wheels[0].setVelocity(left_speed)
        wheels[1].setVelocity(right_speed)
        wheels[2].setVelocity(left_speed)
        wheels[3].setVelocity(right_speed)
    else:
        print("in else")
        wheels[0].setVelocity(0)
        wheels[1].setVelocity(0)
        wheels[2].setVelocity(0)
        wheels[3].setVelocity(0)
        break

# Save collected data to CSV
with open('simulation_data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
