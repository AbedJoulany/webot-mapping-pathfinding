import math
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from controller import Robot

from Sensor import Sensor

TIME_STEP = 64

class Localization:
    def __init__(self, robot):
        self.robot = robot
        self.gps = Sensor(robot, 'global', 'gps', TIME_STEP)
        self.imu = Sensor(robot, 'imu', 'imu', TIME_STEP)
        self.ds_left = Sensor(robot, 'ds_left', 'distance', TIME_STEP)
        self.ds_right = Sensor(robot, 'ds_right', 'distance', TIME_STEP)

    def get_position(self):
        return self.gps.get_value()

    def get_orientation(self):
        return self.imu.get_value()

    def get_left_distance(self):
        return self.ds_left.get_value()

    def get_right_distance(self):
        return self.ds_right.get_value()

