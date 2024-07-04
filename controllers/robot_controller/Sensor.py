import math
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from controller import Robot

class Sensor:
    def __init__(self, robot, name, sensor_type, timestep):
        self.sensor = robot.getDevice(name)
        if sensor_type == 'gps':
            self.sensor.enable(timestep)
        elif sensor_type == 'imu':
            self.sensor.enable(timestep)
        elif sensor_type == 'distance':
            self.sensor.enable(timestep)

    def get_value(self):
        if hasattr(self.sensor, 'getValues'):
            return self.sensor.getValues()
        elif hasattr(self.sensor, 'getValue'):
            return self.sensor.getValue()
        elif hasattr(self.sensor, 'getRollPitchYaw'):
            return self.sensor.getRollPitchYaw()
        else:
            raise AttributeError(f"Sensor {self.sensor.getName()} does not have a getValue, getValues, or getRollPitchYaw method.")

