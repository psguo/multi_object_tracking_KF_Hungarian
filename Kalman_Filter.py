import numpy as np

class Kalman_Filter(object):
    def __init__(self, dt=0.1):
        self.dt = dt

        # # dynamics
        # self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        # self.x = np.zeros((2, 1))
        # self.G = np.array([[self.dt**2/2], [self.dt]])
        # self.P = np.diag((3.0, 3.0))
        # self.Q = np.eye(self.x.shape[0])
        #
        # # observation
        # self.y = np.array([[0], [255]])
        # self.H = np.array([[1, 0], [0, 1]])
        # self.R = np.eye(self.y.shape[0])
        #
        # self.prevResult = np.array([[0], [255]])

        # dynamics
        self.F = np.array([[1.0, self.dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, self.dt], [0.0, 0.0, 0.0, 1.0]])
        self.x = np.zeros((4, 1))
        self.G = np.array([[self.dt ** 2 / 2, 0.0], [self.dt, 0.0], [0.0, self.dt ** 2 / 2], [0.0, self.dt]])
        self.P = np.diag((3.0, 3.0, 3.0, 3.0))
        self.Q = np.eye(self.x.shape[0])

        # observation
        self.y = np.zeros((4, 1))
        self.H = np.eye(self.y.shape[0])
        self.R = np.eye(self.y.shape[0])

    def predict(self, u=0):
        self.x = np.round(np.dot(self.F, self.x) + self.G * u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def correct(self, y):
        self.y = y
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        W = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        return self.update(S, W)

    def update(self, S, W):
        self.x = np.round(self.x + np.dot(W, (self.y - np.dot(self.H, self.x))))
        self.P = self.P - np.dot(W, np.dot(S, W.T))
        return self.x