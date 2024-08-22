import numpy as np


class EKF:
    def __init__(self, dt, state_dim, meas_dim, control_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.control_dim = control_dim
        self.x = np.zeros((state_dim, 1))  # State vector
        self.P = np.eye(state_dim)  # Covariance matrix
        self.Q = np.eye(state_dim) * 0.01  # Process noise covariance
        self.R = np.eye(meas_dim)  # Measurement noise covariance
        self.H = np.eye(meas_dim, state_dim)  # Measurement matrix
        self.F = np.eye(state_dim)  # State transition model

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + u
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def set_F(self, F):
        self.F = F

    def set_H(self, H):
        self.H = H

    def set_Q(self, Q):
        self.Q = Q

    def set_R(self, R):
        self.R = R

    def set_measurement_noise_covariance(self, R):
        self.R = R