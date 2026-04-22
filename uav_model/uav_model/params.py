import numpy as np


class UAVParameters:

    def __init__(self):

        # -------- Physical parameters --------
        self.mass = 1.5
        self.gravity = 9.81

        # inertia matrix
        self.J = np.diag([0.02, 0.02, 0.04])

        # geometry
        self.arm_length = 0.25

        # motor coefficients
        self.kf = 1.0
        self.km = 0.02

        # thrust limits
        self.thrust_min = 0.0
        self.thrust_max = 15.0

        # simulation
        self.dt = 0.002

    # ---------------- Derived quantities ----------------

    def hover_thrust(self):
        return self.mass * self.gravity / 4.0

    def inertia_inv(self):
        return np.linalg.inv(self.J)

    # ---------------- YAML loader ----------------

    def load_from_dict(self, cfg):

        self.mass = cfg.get('mass', self.mass)
        self.gravity = cfg.get('gravity', self.gravity)

        self.arm_length = cfg.get('arm_length', self.arm_length)

        self.kf = cfg.get('kf', self.kf)
        self.km = cfg.get('km', self.km)

        self.thrust_min = cfg.get('thrust_min', self.thrust_min)
        self.thrust_max = cfg.get('thrust_max', self.thrust_max)

        self.dt = cfg.get('dt', self.dt)

        if 'J' in cfg:
            J = np.array(cfg['J'])
            # Accept either a flat 9-element list (row-major) or a 3x3 nested list
            self.J = J.reshape(3, 3)
