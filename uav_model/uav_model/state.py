import numpy as np


class UAVState:
    def __init__(self):

        self.r = np.zeros(3)
        self.v = np.zeros(3)

        # quaternion (w x y z)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.w = np.zeros(3)

    def as_vector(self):
        return np.concatenate([self.r, self.v, self.q, self.w])

    def from_vector(self, x):
        self.r = x[0:3]
        self.v = x[3:6]
        self.q = x[6:10]
        self.w = x[10:13]
        self.normalize()

    def normalize(self):
        self.q = self.q / np.linalg.norm(self.q)

    def copy(self):
        s = UAVState()
        s.from_vector(self.as_vector().copy())
        return s
