import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from PID.pid_controller import PIDController
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt

class BaseRobotController:
    def __init__(self):
        self._robot = Supervisor()
        self._timestep = int(self._robot.getBasicTimeStep())
        self._initialize_devices()

    def _initialize_devices(self):
        self._timestep = 64
        self._epuck_robot = self._robot.getFromDef("e-puck")
        self._rotation_field = self._epuck_robot.getField("rotation")

        self._lidar = self._robot.getDevice("lidar")
        self._lidar.enable(self._timestep)
        self._lidar.enablePointCloud()

        self._gps = self._robot.getDevice("gps")
        self._gps.enable(self._timestep)

        self._compass = self._robot.getDevice("compass")
        self._compass.enable(self._timestep)

        self._camera = self._robot.getDevice('camera')
        #self._camera.enable(self._timestep)

        self._left_motor = self._robot.getDevice("left wheel motor")
        self._right_motor = self._robot.getDevice("right wheel motor")
        self._left_motor.setPosition(float('inf'))
        self._right_motor.setPosition(float('inf'))

        self._left_ps = self._robot.getDevice("right wheel sensor")
        self._left_ps.enable(self._timestep)

        self._right_ps = self._robot.getDevice("left wheel sensor")
        self._right_ps.enable(self._timestep)

        self._keyboard = self._robot.getKeyboard()
        self._keyboard.enable(self._timestep)

        self.distance_pid = PIDController(1.0, 0.01, 0.1)
        self.heading_pid = PIDController(0.1, 0.01, 0.05)

        self._init_constants()
        self._initialize_mapping()

        if self._lidar is None:
            raise ValueError("Lidar device not found. Ensure the lidar device is correctly named and exists in the Webots world file.")

    def _init_constants(self):
        self.wheel_radius = 0.0205
        self.distance_between_wheels = 0.052
        self.wheel_circumference = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circumference / (2 * math.pi)
        self.keys = {key: False for key in ["w", "a", "s", "d", "o", "m", "h", "t", "y","p"]}
        self.key_map = {
            87: 'w', 65: 'a', 83: 's', 68: 'd', 79: 'o',
            77: 'm', 78: 'n', 72: 'h', 84: 't', 89: 'y',80:"p"
        }
        self.manual_control = {"active": True, "count": 0}
        self.path = {"path": 0}
        self.world_size = (2, 2)
        self.map_size = (2, 2)
        self.h_true_pos = np.zeros((3, 0))
        self.h_enco_pos = np.zeros((3, 0))
        self.map = np.zeros((0, 2))
        self.file_path = 'map.csv'
        #self.map_size = 2  # meters
        self.resolution = 0.05  # meter per cell
    
    def _initialize_mapping(self):
        self.ps_values = [0, 0]
        self.dist_values = [0, 0]
        self.last_ps_values = [0, 0]
        self.target = self._robot.getFromDef("target")

    @property
    def robot(self):
        return self._robot

    @property
    def timestep(self):
        return self._timestep

    @property
    def lidar(self):
        return self._lidar

    @property
    def left_motor(self):
        return self._left_motor

    @property
    def right_motor(self):
        return self._right_motor

    @property
    def keyboard(self):
        return self._keyboard

    @property
    def gps(self):
        return self._gps

    @property
    def compass(self):
        return self._compass

    @property
    def camera(self):
        return self._camera

    def set_motor_speeds(self, left_speed, right_speed):
        self._left_motor.setVelocity(left_speed)
        self._right_motor.setVelocity(right_speed)

    def get_gps_position(self):
        return self._gps.getValues()

    def get_compass_heading(self):
        compass_values = self._compass.getValues()
        return compass_values[0], compass_values[2]

    def get_robot_pose_from_webots(self):
        robot_pos = self._epuck_robot.getPosition()
        robot_rot = self._rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)
        return [round(robot_pos[0], 3), round(robot_pos[1], 3), robot_rot_z]

    def get_target_position(self):
        target_pos = self.target.getPosition()
        return np.array([round(target_pos[0], 3), round(target_pos[1], 3)])

    def move_robot_to_waypoints(self, waypoints):
        current_position = self.get_robot_pose_from_webots()
        for waypoint in waypoints:
            self._move_towards_waypoint(current_position, waypoint)
            current_position = self.get_robot_pose_from_webots()
        self.set_motor_speeds(0, 0)
        return True

    def _move_towards_waypoint(self, current_position, target_position):
        def reached_waypoint(curr_pos, target_pos, threshold=0.1):
            return np.linalg.norm(np.array(curr_pos[:2]) - np.array(target_pos[:2])) < threshold

        self.adjust_heading(current_position, target_position)

        while not reached_waypoint(current_position, target_position):
            self.set_motor_speeds(6.28, 6.28)
            self.robot.step(self._timestep)
            current_position = self.get_robot_pose_from_webots()

    def adjust_heading(self, current_pos, target_pos):
        angle_threshold = 2  # Degree threshold to consider the heading aligned
        max_speed = 1  # Maximum speed for rotation

        self.robot.step(self._timestep)
        current_heading = math.degrees(current_pos[2]) % 360
        target_heading = math.degrees(math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])) % 360

        angle_diff = (target_heading - current_heading + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        left_speed, right_speed = (max_speed, -max_speed) if angle_diff < 0 else (-max_speed, max_speed)

        while abs(angle_diff) > angle_threshold:
            self.set_motor_speeds(left_speed, right_speed)
            self.robot.step(self._timestep)

            current_heading = math.degrees(self.get_robot_pose_from_webots()[2]) % 360
            angle_diff = (target_heading - current_heading + 360) % 360
            if angle_diff > 180:
                angle_diff -= 360


    def get_lidar_points(self):
        pointCloud = self.lidar.getRangeImage()
        z = np.zeros((0, 2))
        angle_i = np.zeros((0, 1))

        robot_pose = self.get_robot_pose_from_webots()

        for i in range(len(pointCloud)):
            angle = ((len(pointCloud) - i) * 0.703125 * math.pi / 180) + robot_pose[2] + self.w * 0.1
            angle = (angle + math.pi) % (2 * math.pi) - math.pi

            vx = self.v * math.cos(robot_pose[2])
            vy = self.v * math.sin(robot_pose[2])

            ox = round(math.cos(angle) * pointCloud[i] + robot_pose[0] + vx*0.1, 3)
            oy = round(math.sin(angle) * pointCloud[i] + robot_pose[1] + vy*0.1, 3)

            zi = np.array([ox, oy])
            z = np.vstack((z, zi))
            angle_i = np.vstack((angle_i, angle))

        z_points = np.zeros((0, 2))
        for i in z:
            if not (i[0] == np.inf or i[0] == -np.inf):
                z_points = np.vstack((z_points, i))

        return z, z_points, angle_i, pointCloud


    def odometry(self):
        self.ps_values[0] = self._left_ps.getValue()
        self.ps_values[1] = self._right_ps.getValue()

        dist_values = [
            (self.ps_values[i] - self.last_ps_values[i]) * self.encoder_unit
            for i in range(2)
        ]

        v = (dist_values[0] + dist_values[1]) / 2.0
        w = (dist_values[0] - dist_values[1]) / self.distance_between_wheels

        self.w = w
        self.v = v
        #self.robot_pose_encoder[2] = (
        #    (self.robot_pose_encoder[2] + w) % (2 * math.pi) - math.pi
        #)
        self.robot_pose_encoder[2] = self.robot_pose_encoder[2] + (w * 1)
        self.robot_pose_encoder[2] = ((self.robot_pose_encoder[2] + math.pi) % (2 * math.pi) - math.pi)
        vx = v * math.cos(self.robot_pose_encoder[2])
        vy = v * math.sin(self.robot_pose_encoder[2])
        self.robot_pose_encoder[0] = self.robot_pose_encoder[0] + vx
        self.robot_pose_encoder[1] = self.robot_pose_encoder[1] + vy

        self.last_ps_values = self.ps_values[:]
        return v, w


    def calculate_wheel_speed(self, v, w):
        vL = round(((2 * v) + (w * self.distance_between_wheels)) / 2,3)
        vR = round(((2 * v) - (w * self.distance_between_wheels)) / 2, 3)
        return vL / self.wheel_radius, vR / self.wheel_radius

    def keyboard_control(self):
        keycode = self.keyboard.getKey()

        if keycode in self.key_map:
            key = self.key_map[keycode]
            self.keys = {k: False for k in self.keys}
            self.keys[key] = True

        return keycode


    def update_robot_poses(self, robot_pose):
        self.h_true_pos = np.hstack((self.h_true_pos, robot_pose))
        self.h_enco_pos = np.hstack((self.h_enco_pos, self.robot_pose_encoder))

    def calculate_distance_to_target(self, robot_pose, hedef_pos):
        xd = hedef_pos[0] - robot_pose[0]
        yd = hedef_pos[1] - robot_pose[1]
        return math.hypot(xd, yd)


    def should_update_map(self, current_time, z_points, previous_time, time_threshold):
        return current_time < previous_time or len(z_points) == 0

    def update_map(self, z_points, current_time, previous_time):
        self.map = np.vstack((self.map, z_points))
        return 0

    def save_map(self):
        np.savetxt("map.csv", self.map, delimiter=",")
        print("Map saved to map.csv")



# Example usage
