from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.robot_traj, = self.ax.plot([], [], 'bo-', label="Robot Trajectory")
        self.lidar_points, = self.ax.plot([], [], 'ro', label="Lidar Points")
        self.goal, = self.ax.plot([], [], 'go', label="Goal")
        self.ax.legend()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

    def update(self, robot_pose, lidar_points, path_id):
        self.robot_traj.set_data(robot_pose[0, :], robot_pose[1, :])
        self.lidar_points.set_data(lidar_points[:, 0], lidar_points[:, 1])

        goal_positions = [
            [-0.75, 0.25], [0.25, 0.25], [0.25, 0.75], [-0.75, 0.75],
            [0.25, 0.70], [0.25, 0.25], [-0.75, 0.25], [-0.75, -0.75],
            [-0.25, -0.75], [-0.25, -0.25], [0.75, -0.25], [0.75, 0.75],
            [0.75, -0.25], [-0.25, -0.25], [-0.25, -0.75], [0.75, -0.75]
        ]
        if path_id < len(goal_positions):
            self.goal.set_data(goal_positions[path_id][0], goal_positions[path_id][1])

    def refresh(self):
        plt.pause(0.001)