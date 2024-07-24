import numpy as np

class EKF:
    def __init__(self, dt, state_dim, meas_dim, control_dim=0):
        self.dt = dt
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.control_dim = control_dim

        self.x = np.zeros((state_dim, 1))  # State vector
        self.P = np.eye(state_dim)         # Covariance matrix
        self.F = np.eye(state_dim)         # State transition model
        self.H = np.zeros((meas_dim, state_dim))  # Measurement model
        self.R = np.eye(meas_dim)          # Measurement noise covariance
        self.Q = np.eye(state_dim)         # Process noise covariance

        self.B = np.zeros((state_dim, control_dim))  # Control input model

        #adjust these to get better accuracy
        self.__noise_ax = 9
        self.__noise_ay = 9

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def predict_x(self, u=0):
        """
        Predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you.
        """
        self.x = dot(self.F, self.x) + dot(self.compute_B(), u)

    def predict(self, u=np.zeros((0, 1))):
        # State prediction
        self.predict_x(u)
        # error Covariance prediction
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

                # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z):
        # Linearize the measurement function using Jacobian
        H = self.compute_jacobian_H()
        
        # Kalman gain
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        # State update
        y = z - self.measurement_function(self.x)
        self.x = self.x + np.dot(K, y)
        # Covariance update
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, H)), self.P)

    def set_F(self, F):
        self.F = F

    def set_H(self, H):
        self.H = H

    def set_Q(self, Q):
        self.Q = Q

    def set_R(self, R):
        self.R = R

    def set_B(self, B):
        self.B = B

    def set_state(self, x):
        self.x = x

    def set_covariance(self, P):
        self.P = P

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P

    def measurement_function(self, x):
        # Define the nonlinear measurement function
        return np.array([
            [x[0, 0]],
            [x[1, 0]],
            [x[2, 0]]
        ])

    def compute_jacobian_F(self):
        dt = self.dt
        theta = self.x[2, 0]
        return np.array([
            [1, 0, -dt * self.x[0, 0] * np.sin(theta)],
            [0, 1,  dt * self.x[0, 0] * np.cos(theta)],
            [0, 0, 1]
        ])

    def compute_jacobian_H(self):
        dt = self.dt
        return np.eye(self.meas_dim) * dt



    def compute_B(self):
        dt = self.dt
        return np.array([
            [dt * np.cos(self.x[2, 0]), 0],
            [dt * np.sin(self.x[2, 0]), 0],
            [0, dt]
        ])

    def compute_control_input_model(self):
        dt = self.dt
        return np.array([
            [dt * np.cos(self.x[2, 0]), 0],
            [dt * np.sin(self.x[2, 0]), 0],
            [0, dt]
        ])
