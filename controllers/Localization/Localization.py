from controller import Robot, GPS, InertialUnit, DistanceSensor
import math
import numpy as np

# Constants
TIME_STEP = 64

class Localization:
    def __init__(self):
        self.robot = Robot()

        # Initialize sensors
        self.gps = self.robot.getDevice('global')
        self.gps.enable(TIME_STEP)
        self.imu = self.robot.getDevice('imu')
        self.imu.enable(TIME_STEP)
        self.ds_left = self.robot.getDevice('ds_left')
        self.ds_right = self.robot.getDevice('ds_right')
        self.ds_left.enable(TIME_STEP)
        self.ds_right.enable(TIME_STEP)

        # Initialize motors
        self.motors = []
        motor_names = ['bk_right_wheel', 'fr_right_wheel', 'bk_left_wheel', 'fr_left_wheel']
        for name in motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(float('inf'))  # Set to velocity control
            motor.setVelocity(0.0)
            self.motors.append(motor)
        
        # Localization variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def set_motor_speeds(self, left_speed, right_speed):
        for i in range(2):
            self.motors[i].setVelocity(right_speed)
        for i in range(2, 4):
            self.motors[i].setVelocity(left_speed)

    def get_gps_position(self):
        return self.gps.getValues()

    def get_imu_orientation(self):
        return self.imu.getRollPitchYaw()

    def to_euler_angles(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            # Get sensor readings
            gps_position = self.get_gps_position()
            imu_orientation = self.get_imu_orientation()
            ds_left_value = self.ds_left.getValue()
            ds_right_value = self.ds_right.getValue()

            # Update localization data
            self.current_x, self.current_y, self.current_z = gps_position
            self.roll, self.pitch, self.yaw = imu_orientation

            # Print localization data
            print(f"GPS: {gps_position}")
            print(f"IMU: Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}")

            # Simple control logic (example: move forward)
            self.set_motor_speeds(1.0, 1.0)

# Create the Localization instance and run the localization
localization = Localization()
localization.run()
