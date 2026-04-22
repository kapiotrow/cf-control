class RK4Integrator:

    def __init__(self, dynamics_function):
        self.f = dynamics_function

    def step(self, state_vec, control, params, dt):

        k1 = self.f(state_vec, control, params)
        k2 = self.f(state_vec + 0.5 * dt * k1, control, params)
        k3 = self.f(state_vec + 0.5 * dt * k2, control, params)
        k4 = self.f(state_vec + dt * k3, control, params)

        next_state = state_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return next_state
