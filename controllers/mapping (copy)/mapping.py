# main.py

from robot_controller import RobotController
from manual import manual_navigation
import cv2
"""
def main():
    controller = RobotController()
    robot = controller.robot
    timestep = controller.timestep
    lidar = controller.lidar
    left_motor = controller.left_motor
    right_motor = controller.right_motor

    mode = 2  # Change mode as needed

    if mode == 1:
        print("1. Autonomous Navigation")
        autonomous_navigation(robot, timestep, lidar, left_motor, right_motor)

    elif mode == 2:
        print("2. Manual Navigation (use WASD keys)")
        manual_navigation(controller)

    cv2.destroyAllWindows()  # Close OpenCV windows when done

if __name__ == "__main__":
    main()"""


if __name__ == "__main__":
    robot_controller = RobotController()
    robot_controller.run()