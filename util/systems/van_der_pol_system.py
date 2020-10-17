import casadi as cs
import numpy as np


class CasadiSimulator:
    def __init__(self):
        # Time
        self.t = cs.MX.sym('t')

        # States
        self.x1 = cs.MX.sym('x1')
        self.x2 = cs.MX.sym('x2')

        self.states = cs.horzcat(self.x1, self.x2)

        # Control
        self.u = cs.MX.sym('u')

        self.ode = cs.horzcat((1 - self.x2 ** 2) * self.x1 - self.x2 + self.u,
                              self.x1)
        self.dae = {'x': cs.transpose(self.states), 'p': self.u, 't': self.t, 'ode': cs.transpose(self.ode)}

    def run(self, np_t, np_u, np_ic, output_t0=True):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': output_t0})
        sol = integrator(x0=np.reshape(np_ic, self.states.shape),
                         p=np.reshape(np_u, self.u.shape))
        return np.transpose(np.array(sol['xf']))

    def get_runge_kutta(self, T, runge_kutta_steps=4):
        DT = T / runge_kutta_steps
        f = cs.Function('f', [self.u, self.states], [self.ode])

        y0 = cs.MX.sym('y0', 1, 2)

        u = cs.MX.sym('u')
        y = y0

        for j in range(runge_kutta_steps):
            k1 = f(u, y)
            k2 = f(u, y + DT / 2 * k1)
            k3 = f(u, y + DT / 2 * k2)
            k4 = f(u, y + DT * k3)
            y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [u, y0], [y], ['u', 'y0'], ['yf'])
