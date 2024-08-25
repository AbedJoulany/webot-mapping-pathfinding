from controller import Motor, Camera, CameraRecognitionObject, Supervisor
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import math
import scipy.cluster.hierarchy as hcluster
import time

# Helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findLines(dists):
    fov = 3.14
    theta_is = np.linspace(0, fov, len(dists))
    theta_js = np.linspace(0, fov, 90)
    dists[dists == inf] = 0
    max_dist = np.amax(dists)
    dist_js = np.linspace(-max_dist, max_dist, 100)
    accum = np.zeros((len(dist_js), len(theta_js)))

    for i in range(len(dists)):
        theta_i = theta_is[i]
        for j in range(len(theta_js)):
            dj = dists[i] * math.cos(theta_js[j] - theta_i)
            dj_id = find_nearest(dist_js, dj)
            accum[dj_id, j] += 1

    rows, cols = np.where(accum >= 70)
    final = np.array([[dist_js[rows[i]], theta_js[cols[i]]] for i in range(len(rows))]).T
    return final

def EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt):
    x = x_hat_t[0] + u[0] * np.cos(x_hat_t[2]) * dt
    y = x_hat_t[1] + u[0] * np.sin(x_hat_t[2]) * dt
    theta = x_hat_t[2] + u[1] * dt
    x_hat_t = np.array([x, y, theta])
    phi = np.array([[1, 0, -u[0] * np.sin(x_hat_t[2]) * dt], [0, 1, u[0] * np.cos(x_hat_t[2]) * dt], [0, 0, 1]])
    G = np.array([[np.cos(x_hat_t[2]) * dt, 0], [np.sin(x_hat_t[2]) * dt, 0], [0, dt]])
    Sigma_x_t = phi @ Sigma_x_t @ phi.T + G @ Sigma_n @ G.T
    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, Sigma_x_t, z, Sigma_m, G_p_L, dt):
    pos = x_hat_t[:2]
    theta = x_hat_t[2]
    z_hat = rotMat(theta).T @ (G_p_L[:2] - pos)
    r = z - z_hat
    H = np.hstack((-rotMat(theta).T, (-rotMat(theta).T @ np.array([[0, -1], [1, 0]]) @ (G_p_L[:2] - pos)).reshape((2, 1))))
    S = H @ Sigma_x_t @ H.T + Sigma_m
    K = Sigma_x_t @ H.T @ np.linalg.inv(S)
    x_hat_t = x_hat_t + K @ r
    Sigma_x_t = Sigma_x_t - K @ H @ Sigma_x_t
    return x_hat_t, Sigma_x_t

def posOfImgToBearing(x, w, fov):
    d = (0.5 * w) / np.tan(0.5 * fov)
    return np.arctan2(0.5 * w - x, d)

def initialize_robot():
    robot = Supervisor()
    camera = robot.getDevice('camera')
    camera.enable(1)
    if camera.hasRecognition():
        camera.recognitionEnable(1)
        camera.enableRecognitionSegmentation()
    else:
        print("Your camera does not have recognition")
    
    timestep = int(robot.getBasicTimeStep())
    leftMotor = robot.getDevice('left wheel motor')
    rightMotor = robot.getDevice('right wheel motor')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))
    rightMotor.setVelocity(0)
    leftMotor.setVelocity(0)
    
    left_ps = robot.getDevice('left_wheel_sensor')
    left_ps.enable(timestep)
    right_ps = robot.getDevice('right_wheel_sensor')
    right_ps.enable(timestep)
    
    lidar = robot.getDevice('lidar')
    lidar.enable(1)
    lidar.enablePointCloud()
    
    return robot, camera, leftMotor, rightMotor, left_ps, right_ps, lidar, timestep

def update_position(robotNode, vel, dt):
    x_s_est = vel[0] * dt
    y_s_est = vel[1] * dt
    orient = vel[5] * dt
    if orient < 0:
        orient += 2 * np.pi
    return x_s_est, y_s_est, orient

def process_lidar_data(lidar, thresh):
    lidar_dat = np.array(lidar.getRangeImage())
    lines = findLines(lidar_dat)
    clusters = hcluster.fclusterdata(lines, thresh, criterion="distance")
    centroids = [np.mean(cluster, axis=0) for cluster in np.split(lines, np.unique(clusters, return_index=True)[1][1:])]
    return centroids

def main_loop(robot, camera, leftMotor, rightMotor, left_ps, right_ps, lidar, timestep):
    robotNode = robot.getFromDef("e-puck")
    mode = 0
    mode_dir = -1
    flag = False
    flag3 = False
    wall_thresh = 0.1
    goal_pos = [-2.9, 0.747, 0.0]
    dt = 0.032

    while robot.step(timestep) != -1:
        start_pos = robotNode.getPosition()
        start_orient = robotNode.getOrientation()
        vel = robotNode.getVelocity()
        theta_s = np.arctan2(start_orient[3], start_orient[0])
        if theta_s < 0:
            theta_s += 2 * np.pi

        x_s_est, y_s_est, orient = update_position(robotNode, vel, dt)
        centroids = process_lidar_data(lidar, 0.05)

        # Mode-based actions
        if mode == 0:
            # Default mode: turn towards goal, then go forward
            handle_mode_0(theta_s, goal_pos, x_s_est, y_s_est, leftMotor, rightMotor)
        elif mode == 1:
            # Wall turn mode
            mode, mode_dir = handle_mode_1(theta_s, centroids, wall_thresh, lidar, leftMotor, rightMotor, mode_dir)
        elif mode == 2:
            # Wall forward mode
            mode, flag = handle_mode_2(centroids, leftMotor, rightMotor, flag)
        elif mode == 3:
            # SLAM mode
            flag3, mode = handle_mode_3(flag3, camera, robotNode, vel, dt, leftMotor, rightMotor, centroids)

        objs = camera.getRecognitionObjects()
        if len(objs) >= 3 and mode != 3 and not flag3:
            rightMotor.setVelocity(6)
            leftMotor.setVelocity(6)
            mode = 3

def handle_mode_0(theta_s, goal_pos, x_s_est, y_s_est, leftMotor, rightMotor):
    y = goal_pos[1] - y_s_est
    x = goal_pos[0] - x_s_est
    theta = np.arctan2(y, x) % (2 * np.pi)
    diff = theta - theta_s
    if abs(diff) > 0.05:
        leftMotor.setVelocity(-1.2 if diff > 0 else 1.2)
        rightMotor.setVelocity(1.2 if diff > 0 else -1.2)
    else:
        leftMotor.setVelocity(max_vel)
        rightMotor.setVelocity(max_vel)

def handle_mode_1(theta_s, centroids, wall_thresh, lidar, leftMotor, rightMotor, mode_dir):
    for centroid in centroids:
        if abs(centroid[0]) <= wall_thresh and abs(centroid[0]) != 0.0:
            zero = np.mean(lidar.getRangeImage()[0:30])
            end = np.mean(lidar.getRangeImage()[480:511])
            if zero < end and mode_dir == -1:
                mode_dir = 1
            leftMotor.setVelocity(1 if mode_dir == 1 else -1)
            rightMotor.setVelocity(-1 if mode_dir == 1 else 1)
            return 1, mode_dir
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)
    return 2, mode_dir

def handle_mode_2(centroids, leftMotor, rightMotor, flag):
    leftMotor.setVelocity(max_vel)
    rightMotor.setVelocity(max_vel)
    for centroid in centroids:
        if abs(centroid[0]) <= wall_thresh and abs(centroid[0]) != 0.0:
            return 1, flag
    return 2, True

def handle_mode_3(flag3, camera, robotNode, vel, dt, leftMotor, rightMotor, centroids):
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)
    flag3 = True
    mode = 3
    print('Handle SLAM mode logic here')
    return flag3, mode

# Constants and initializations
max_vel = 6.28
wheelRadius = 0.0205
axleLength = 0.053  # should be 56-57mm in reality

# Main execution
if __name__ == '__main__':
    robot, camera, leftMotor, rightMotor, left_ps, right_ps, lidar, timestep = initialize_robot()
    main_loop(robot, camera, leftMotor, rightMotor, left_ps, right_ps, lidar, timestep)
