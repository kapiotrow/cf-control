import numpy as np


class QuadMotorModel:
    def __init__(self, arm_length, kf, km, thrust_min=0.0, thrust_max=15.0):
        self.L = arm_length
        self.kf = kf
        self.km = km
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max

    def clamp(self, thrusts):
        return np.clip(thrusts, self.thrust_min, self.thrust_max)

    def map_to_forces(self, thrust_cmd):

        thrust_cmd = self.clamp(thrust_cmd)

        T1, T2, T3, T4 = thrust_cmd

        # Total thrust
        T = self.kf * (T1 + T2 + T3 + T4)

        # Body torques
        tau_x = self.L * self.kf * (T2 - T4)
        tau_y = self.L * self.kf * (T3 - T1)
        tau_z = self.km * (T1 - T2 + T3 - T4)

        tau = np.array([tau_x, tau_y, tau_z])

        return T, tau
