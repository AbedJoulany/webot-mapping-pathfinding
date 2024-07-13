import math
import numpy as np
from controller import Supervisor, Keyboard, Lidar, GPS
from visualize_grid import create_occupancy_grid
from matplotlib import pyplot as plt
from rrt_star import RRTStar


show_animation = True

# Usage example
file_path = 'map.csv'
map_size = 2  # meters
resolution = 0.001  # meter per cell

class RobotController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._robot = Supervisor()
            cls._instance._timestep = int(cls._instance._robot.getBasicTimeStep())
            cls._instance._initialize_devices()
        return cls._instance

    def _initialize_devices(self):
        # e-puck
        self._epuckRobot = self._robot.getFromDef("e-puck")
        self._rotation_field = self._epuckRobot.getField("rotation")
        
        self._lidar = self._robot.getDevice("lidar")
        self._lidar.enable(self._timestep)
        self._lidar.enablePointCloud()
        
        self._gps = self._robot.getDevice("gps")
        self._gps.enable(self._timestep)
        
        # Compass
        self._compass = self._robot.getDevice("compass")
        self._compass.enable(self._timestep)

        self._left_motor = self._robot.getDevice("left wheel motor")
        self._right_motor = self._robot.getDevice("right wheel motor")
        self._left_motor.setPosition(float('inf'))
        self._right_motor.setPosition(float('inf'))
        
        self.left_ps = self._robot.getDevice("right wheel sensor")
        self.left_ps.enable(self._timestep)

        self.right_ps = self._robot.getDevice("left wheel sensor")
        self.right_ps.enable(self._timestep)
        
        self._keyboard = self._robot.getKeyboard()
        self._keyboard.enable(self._timestep)

        self.ps_values = [0, 0]
        self.dist_values = [0, 0]

        self.wheel_radius = 0.0205
        self.distance_between_wheels = 0.052
        self.wheel_circumference = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circumference / (2 * 3.14)

        self.robot_pose_encoder = self.get_robot_pose_from_webots()
        self.last_ps_values = [0, 0]

        self.keys = {key: False for key in ["w", "a", "s", "d", "o", "m", "h", "t", "y"]}
        self.manual_control = {"active": True, "count": 0}
        self.path = {"path": 0}

        self.hTruePos = np.zeros((3, 0))
        self.hEncoPos = np.zeros((3, 0))

        self.map = np.zeros((0, 2))  # Initialize the map to store lidar points

        self.hedef = self._robot.getFromDef("target")  # Adding target initialization

        self.w = 0
        self.v = 0
        
        self.world_size = (2,2)
        self.map_size = (2,2)    

        if self._lidar is None:
            raise ValueError("Lidar device not found. Ensure the lidar device is correctly named and exists in the Webots world file.")


    def save_map(self):
        np.savetxt("map.csv", self.map, delimiter=",")
        print("Map saved to map.csv")
        

    def world2map(self, gps_position, world_size, floor_size):
        x = y = z = 0
        if len(gps_position) == 3 :
            x, y, z = gps_position
        else:
            x, y= gps_position
        map_x = (x + world_size[0] / 2) / floor_size
        map_y = (y + world_size[1] / 2) / floor_size

        return map_x, map_y

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

    def set_motor_speeds(self, left_speed, right_speed):
        self._left_motor.setVelocity(left_speed)
        self._right_motor.setVelocity(right_speed)

    def get_gps_position(self):
        return self._gps.getValues()

    def get_compass_heading(self):
        compass_values = self._compass.getValues()
        heading = (compass_values[0], compass_values[2])
        return heading


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

    def get_robot_pose_from_webots(self):
        robot_pos = self._epuckRobot.getPosition()
        robot_rot = self._rotation_field.getSFRotation()
        axis_z = robot_rot[2]
        robot_rot_z = round(robot_rot[3], 3) * (-1 if axis_z < 0 else 1)

        return np.array([
            [round(robot_pos[0], 3)],
            [round(robot_pos[1], 3)],
            [robot_rot_z],
        ])

    def odometry(self):
        self.ps_values[0] = self.left_ps.getValue()
        self.ps_values[1] = self.right_ps.getValue()

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
        self.robot_pose_encoder[2] = ((self.robot_pose_encoder[2, 0] + math.pi) % (2 * math.pi) - math.pi)
        vx = v * math.cos(self.robot_pose_encoder[2])
        vy = v * math.sin(self.robot_pose_encoder[2])
        self.robot_pose_encoder[0] = self.robot_pose_encoder[0, 0] + vx
        self.robot_pose_encoder[1] = self.robot_pose_encoder[1, 0] + vy

        self.last_ps_values = self.ps_values[:]
        return v, w

    def calculate_wheel_speed(self, v, w):
        vL = round(((2 * v) + (w * self.distance_between_wheels)) / 2,3)
        vR = round(((2 * v) - (w * self.distance_between_wheels)) / 2, 3)
        return vL / self.wheel_radius, vR / self.wheel_radius

    def keyboard_control(self):
        keycode = self.keyboard.getKey()
        key_map = {
            87: 'w', 65: 'a', 83: 's', 68: 'd', 79: 'o',
            77: 'm', 78: 'n', 72: 'h', 84: 't', 89: 'y'
        }

        if keycode in key_map:
            key = key_map[keycode]
            self.keys = {k: False for k in self.keys}
            self.keys[key] = True

        return keycode



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
                #self.save_map()
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
            elif self.keys["p"]:
                self.save_map()
        else:
            if is_goal or current_time < 3:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
            else:
                self.left_motor.setVelocity(vL)
                self.right_motor.setVelocity(vR)

            hedef_path = [
                [-0.75, 0.25, 0.01], [0.25, 0.25, 0.01], [0.25, 0.75, 0.01], [-0.75, 0.75, 0.01],
                [0.25, 0.70, 0.01], [0.25, 0.25, 0.01], [-0.75, 0.25, 0.01], [-0.75, -0.75, 0.01],
                [-0.25, -0.75, 0.01], [-0.25, -0.25, 0.01], [0.75, -0.25, 0.01], [0.75, 0.75, 0.01],
                [0.75, -0.25, 0.01], [-0.25, -0.25, 0.01], [-0.25, -0.75, 0.01], [0.75, -0.75, 0.01]
            ]

            if (self.keys["h"] or d <= 0.15) and is_goal:
                hedef_trans = self.hedef.getField("translation")
                hedef_trans.setSFVec3f(hedef_path[self.path["path"]])

                if self.path["path"] < len(hedef_path) - 1:
                    self.path["path"] += 1

        return self.path["path"],d

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
            if keycode == 27:  # Escape key
                self.save_map()
                break

            v, w = self.odometry()
            #vL, vR = self.calculate_wheel_speed(v, w)
            #path_id = self.robot_control(vL, vR, is_goal, 0.2,path, current_time)
            z, z_points, angle_i, pointCloud = self.get_lidar_points()

            robot_pose = self.get_robot_pose_from_webots()


            #self.update_robot_poses(robot_pose)
            self.hEncoP = np.hstack((self.hEncoPos, self.robot_pose_encoder))

            if self.should_update_map(time, z_points,previous_time, time_threshold ):
                previous_time = self.update_map(z_points, current_time, previous_time)
            else:
                previous_time += time_threshold
                print("previous_time: ", round(previous_time, 3))

            hedef_pos = self.get_target_position()
            distance_path = self.calculate_distance_to_target(robot_pose, hedef_pos)
            #print("Target distance:: ", distance_path)

            if self.is_goal_reached(distance_path):
                print("Target location reached")
                is_goal = True
                

            vL, vR = self.calculate_wheel_speed(v, w)
            nPath, dist = self.robot_control(vL, vR, is_goal, distance_path, path, current_time)

            # self.update_robot_poses(robot_pose)
            self.hTruePos = np.hstack((self.hTruePos, robot_pose))

            #print("Actual robot positions saved\n")

            if show_animation:
                self.animate_map(robot_pose, hedef_pos, z)

            previous_time = current_time

    def update_robot_poses(self, robot_pose):
        self.hTruePos = np.hstack((self.hTruePos, robot_pose))
        self.hEncoPos = np.hstack((self.hEncoPos, self.robot_pose_encoder))

    def should_update_map(self, current_time, z_points, previous_time, time_threshold):
        return current_time < previous_time or len(z_points) == 0

    def update_map(self, z_points, current_time, previous_time):
        self.map = np.vstack((self.map, z_points))
        #np.savetxt("map.csv", self.map, delimiter=",")
        return 0

    def get_target_position(self):
        hedef_pos = self.hedef.getPosition()
        return np.array([
            round(hedef_pos[0], 3),
            round(hedef_pos[1], 3)
        ])

    def calculate_distance_to_target(self, robot_pose, hedef_pos):
        xd = hedef_pos[0] - robot_pose[0]
        yd = hedef_pos[1] - robot_pose[1]
        return math.hypot(xd, yd)

    def is_goal_reached(self, distance_path):
        return distance_path <= 0.15

    def animate_map(self, robot_pose, hedef_pos, z):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

        plt.plot(self.map[:, 0], self.map[:, 1], ".b", label="map")
        plt.plot(self.hTruePos[0, :], self.hTruePos[1, :], "-b")
        plt.plot(self.hEncoPos[0, :], self.hEncoPos[1, :], "-k")
        plt.plot(z[:, 0], z[:, 1], ".m", label="instant lidar")

        plt.arrow(robot_pose[0, 0], robot_pose[1, 0], 0.05 * math.cos(robot_pose[2]),
                0.05 * math.sin(robot_pose[2]), head_length=0.07, head_width=0.07)
        plt.arrow(self.robot_pose_encoder[0, 0], self.robot_pose_encoder[1, 0], 0.05 * math.cos(self.robot_pose_encoder[2]),
                0.05 * math.sin(self.robot_pose_encoder[2]), head_length=0.07, head_width=0.07, color="k")


        #print("true pos:" ,robot_pose)

        #print("robot_pose_encoder:" ,self.robot_pose_encoder)

        plt.plot(hedef_pos[0], hedef_pos[1], "xg")

        plt.title("Webots_Lidar_Mapping")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        
    def run_rrt_star(self, start, goal, obstacle_list, expand_dis=0.5, path_resolution=0.1, goal_sample_rate=5, max_iter=500):
        
        rand_area = [0, 4]
        obstacle_list = self.obstacles
        rrt_star = RRTStar(start=start, goal=goal, obstacle_list=obstacle_list, rand_area=rand_area, expand_dis=expand_dis, path_resolution=path_resolution, goal_sample_rate=goal_sample_rate, max_iter=max_iter)
        path = rrt_star.planning()
        return path
    

    def handle_keyboard(self, key):
        key_char = chr(key).lower()
        if key_char in self.keys:
            self.keys[key_char] = not self.keys[key_char]
            self.manual_control["active"] = self.keys[key_char]
            print(f"Key {key_char} pressed: {self.keys[key_char]}")
    
    def navigate_to_target(self, path):
        for target_pos in path:
            current_pos = self.get_gps_position()
            distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos[:2]))
            angle_to_target = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
            
            heading = self.get_compass_heading()
            current_heading = math.atan2(heading[1], heading[0])
            
            angle_difference = angle_to_target - current_heading
            angle_difference = (angle_difference + math.pi) % (2 * math.pi) - math.pi
            
            if distance > 0.1:
                if abs(angle_difference) > 0.1:
                    vL, vR = self.calculate_wheel_speed(0, 0.5 if angle_difference > 0 else -0.5)
                else:
                    vL, vR = self.calculate_wheel_speed(5.0, 0)
            else:
                vL, vR = self.calculate_wheel_speed(0, 0)
                
            self.set_motor_speeds(vL, vR)


    def run2(self):

        occupancy_grid = create_occupancy_grid(file_path, map_size, resolution)


        while self.robot.step(self.timestep) != -1:
            key = self.keyboard.getKey()
            while self.keyboard.getKey() != -1:
                pass
            if key != -1:
                self.handle_keyboard(key)

            z, z_points, angle_i, pointCloud = self.get_lidar_points()
            map_x, map_y = self.world2map(self.get_gps_position(), (2, 2), 0.001)

            # Use a set to store unique obstacles
            unique_obstacles = set()

            for point in z_points:
                point_x, point_y = self.world2map(point, (2, 2), 0.001)
                if 0 <= point_x < self.map_size[0] and 0 <= point_y < self.map_size[1]:
                    rounded_point = (round(point_x, 3), round(point_y, 3))
                    unique_obstacles.add(rounded_point)  # Add to the set to ensure uniqueness

            self.obstacles = list(unique_obstacles)  # Convert set back to list for further processing

            robot_pos = self.get_gps_position()[:2]
            goal_pos = [2, 2]  # Example goal position, adjust as needed

            path = self.run_rrt_star(start=robot_pos, goal=goal_pos, obstacle_list=self.obstacles)

            if path:
                self.navigate_to_target(path)
            else:
                print("No path found")

if __name__ == "__main__":
    controller = RobotController()

    controller.run2()
