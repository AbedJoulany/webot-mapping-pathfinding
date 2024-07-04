import math
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from controller import Robot


class Motor:
    def __init__(self, robot, name):
        self.motor = robot.getDevice(name)
        self.motor.setPosition(float('inf'))  # Set to velocity control
        self.motor.setVelocity(0.0)

    def set_velocity(self, velocity):
        self.motor.setVelocity(velocity)

