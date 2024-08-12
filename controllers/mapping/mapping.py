# main.py

#from robot_controller import RobotController
from path_finding_robot_ekf import PathFindingRobotEKFController
from data_collect import DataCollectorRobotController
from path_finding_robot import PathFindingRobotController

from manual import manual_navigation
from manual_mapping_robot import ManualMappingController


if __name__ == "__main__":

    mode = 1

    if mode == 0:
        manual_mapping_controller = ManualMappingController()

        manual_mapping_controller.run()


    if mode  == 1:
        # Create instance of RobotController
        path_finding_robot_controller = PathFindingRobotController()
        # Example start and goal positions
        start_position = path_finding_robot_controller.get_robot_pose_from_webots()
        x,y = path_finding_robot_controller.get_target_position()
        goal_position = (x,y,1.0)
        # Plan and follow path
        path_finding_robot_controller.plan_and_follow_path(start_position, goal_position)


    if mode == 2:
        path_finding_robot_controller = PathFindingRobotEKFController()
                # Example start and goal positions
        start_position = path_finding_robot_controller.get_robot_pose_from_webots()
        x,y = path_finding_robot_controller.get_target_position()
        goal_position = (x,y,1.0)
        # Plan and follow path
        path_finding_robot_controller.plan_and_follow_path(start_position, goal_position)
        print("finished ````````````````")


    if mode == 3:
        robot = DataCollectorRobotController()
        robot.move_random()






















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
