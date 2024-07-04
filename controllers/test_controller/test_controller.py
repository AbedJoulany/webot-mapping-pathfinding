from controller import Robot, Supervisor, Camera, GPS, InertialUnit, DistanceSensor, Keyboard

TIME_STEP = 32

# wheelsNames = ['fr_left_wheel', 'fr_right_wheel', 'bk_left_wheel',  'bk_right_wheel']


class Localization:
    def __init__(self, robot, supervisor):
        self.robot = robot
        self.supervisor = supervisor
        self.gps = robot.getDevice("global")
        self.imu = robot.getDevice("imu")
        self.gps.enable(TIME_STEP)
        self.imu.enable(TIME_STEP)

    def update(self):
        gps_values = self.gps.getValues()
        imu_values = self.imu.getQuaternion()
        self.publish_base_link(gps_values, imu_values)

    def publish_base_link(self, gps_values, imu_values):
        base_link = self.supervisor.getFromDef("base_link")
        if base_link:
            translation = base_link.getField("translation")
            translation.setSFVec3f(gps_values)
            rotation = base_link.getField("rotation")
            rotation.setSFRotation(imu_values)

class StaticCamera:
    def __init__(self, robot):
        self.robot = robot
        self.linear_motor = robot.getDevice("linear")
        self.rotation_motor = robot.getDevice("RM")
        self.linear_motor.setPosition(float('inf'))
        self.linear_motor.setVelocity(0)
        self.rotation_motor.setPosition(float('inf'))
        self.rotation_motor.setVelocity(0)

    def update(self):
        linear_value = self.linear_motor.getTargetPosition()
        rotation_value = self.rotation_motor.getTargetPosition()
        # Publish linear and rotation values for use by the supervisor controller
        print(f"linear:{linear_value}")
        print(f"rotation:{rotation_value}")

class SensorEnable:
    def __init__(self, robot):
        self.robot = robot
        self.keyboard = robot.getKeyboard()
        self.keyboard.enable(TIME_STEP)
        self.wheels = [robot.getDevice(name) for name in ['fr_left_wheel', 'fr_right_wheel', 'bk_left_wheel',  'bk_right_wheel']

]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)

    def update(self):
        key = self.keyboard.getKey()
        if key != -1:
            self.teleop(key)

    def teleop(self, key):
        velocity = 2.0  # Example speed value
        if key == Keyboard.UP:
            for wheel in self.wheels:
                wheel.setVelocity(velocity)
        elif key == Keyboard.DOWN:
            for wheel in self.wheels:
                wheel.setVelocity(-velocity)
        elif key == Keyboard.LEFT:
            self.wheels[0].setVelocity(-velocity)
            self.wheels[1].setVelocity(velocity)
            self.wheels[2].setVelocity(-velocity)
            self.wheels[3].setVelocity(velocity)
        elif key == Keyboard.RIGHT:
            self.wheels[0].setVelocity(velocity)
            self.wheels[1].setVelocity(-velocity)
            self.wheels[2].setVelocity(velocity)
            self.wheels[3].setVelocity(-velocity)
        else:
            for wheel in self.wheels:
                wheel.setVelocity(0)

class MainController:
    def __init__(self):
        self.robot = Robot()
        self.sensor_enable = SensorEnable(self.robot)
        self.static_camera = StaticCamera(self.robot)

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            self.sensor_enable.update()
            self.static_camera.update()

if __name__ == "__main__":
    main_controller = MainController()
    main_controller.run()
class MainController:
    def __init__(self):
        self.robot = Robot()
        self.supervisor = Supervisor()
        self.localization = Localization(self.robot, self.supervisor)
        self.sensor_enable = SensorEnable(self.robot, self.supervisor)
        self.static_camera = StaticCamera(self.robot, self.supervisor)

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            self.localization.update()
            self.sensor_enable.update()
            self.static_camera.update()

if __name__ == "__main__":
    main_controller = MainController()
    main_controller.run()