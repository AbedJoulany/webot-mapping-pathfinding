import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt
from base_robot_controller import BaseRobotController

show_animation = True

class ManualMappingController(BaseRobotController):


    def __init__(self, robot_name="e-puck"):
        super().__init__(robot_name)

    def robot_control(self, vL, vR, is_goal, d,path_id, current_time):
        if self.keys["o"]:
            self.manual_control["active"] = not self.manual_control["active"]
            self.manual_control["count"] += 1

        if self.manual_control["active"]:
            if self.keys["a"]:
                self.left_motor.setVelocity(-1)
                self.right_motor.setVelocity(1)
            elif self.keys["d"]:
                self.left_motor.setVelocity(1)
                self.right_motor.setVelocity(-1)
            elif self.keys["w"]:
                self.left_motor.setVelocity(6.28)
                self.right_motor.setVelocity(6.28)
            elif self.keys["s"]:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
            elif self.keys["p"]:
                self.save_map()
            else:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)

        return self.path["path"],d

    def get_robot_pose_from_webots(self):
        robot_pos = self._epuck_robot.getPosition()
        robot_rot = self._rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)

        return np.array([
            [round(robot_pos[0], 3)],
            [round(robot_pos[1], 3)],
            [robot_rot_z],
        ])

    def get_lidar_points(self):
        pointCloud = self.lidar.getRangeImage()
        z = np.zeros((0, 2))
        angle_i = np.zeros((0, 1))

        robot_pose = self.get_robot_pose_from_webots()

        for i in range(len(pointCloud)):
            angle = ((len(pointCloud) - i) * 0.703125 * math.pi / 180) + robot_pose[2, 0] + self.w * 0.1
            angle = (angle + math.pi) % (2 * math.pi) - math.pi

            vx = self.v * math.cos(robot_pose[2])
            vy = self.v * math.sin(robot_pose[2])

            ox = round(math.cos(angle) * pointCloud[i] + robot_pose[0, 0] + vx*0.1, 3)
            oy = round(math.sin(angle) * pointCloud[i] + robot_pose[1, 0] + vy*0.1, 3)

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


    def animate_map(self, robot_pose, hedef_pos, z):
        plt.clf()

        plt.plot(self.map[:, 0], self.map[:, 1], ".b", label="map")
        plt.plot(self.h_true_pos[0, :], self.h_true_pos[1, :], "-b")
        plt.plot(self.h_enco_pos[0, :], self.h_enco_pos[1, :], "-k")
        plt.plot(z[:, 0], z[:, 1], ".m", label="instant lidar")

        plt.arrow(robot_pose[0, 0], robot_pose[1, 0], 0.05 * math.cos(robot_pose[2]),
                0.05 * math.sin(robot_pose[2]), head_length=0.07, head_width=0.07)
        plt.arrow(self.robot_pose_encoder[0, 0], self.robot_pose_encoder[1, 0], 0.05 * math.cos(self.robot_pose_encoder[2]),
                0.05 * math.sin(self.robot_pose_encoder[2]), head_length=0.07, head_width=0.07, color="k")

        plt.plot(hedef_pos[0], hedef_pos[1], "xg")

        plt.title("Webots_Lidar_Mapping")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)



    def run(self):
        print("starting...")
        current_time = 0

        previous_time = 0 
        time = 4 
        is_goal = False
        time_threshold = 0.064 

        path = {
            "path": 0
        }
        self.robot_pose_encoder = self.get_robot_pose_from_webots()

        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()

            keycode = self.keyboard_control()


            v, w = self.odometry()
            z, z_points, angle_i, pointCloud = self.get_lidar_points()

            robot_pose = self.get_robot_pose_from_webots()

            self.h_enco_pos = np.hstack((self.h_enco_pos, self.robot_pose_encoder))

            if self.should_update_map(time, z_points,previous_time, time_threshold ):
                previous_time = self.update_map(z_points, current_time, previous_time)
            else:
                previous_time += time_threshold
                print("previous_time: ", round(previous_time, 3))

            hedef_pos = self.get_target_position()
            distance_path = self.calculate_distance_to_target(robot_pose, hedef_pos)

            vL, vR = self.calculate_wheel_speed(v, w)
            nPath, dist = self.robot_control(vL, vR, is_goal, distance_path, path, current_time)

            self.h_true_pos = np.hstack((self.h_true_pos, robot_pose))

            if show_animation:
                self.animate_map(robot_pose, hedef_pos, z)

            previous_time = current_time
