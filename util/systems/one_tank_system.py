import casadi as cs
import numpy as np


class CasadiSimulator:
    def __init__(self, sys_params):
        # Time
        self.t = cs.MX.sym('t')  # [s]

        # State
        self.h = cs.MX.sym('h')  # [cm]

        # Control
        self.v = cs.MX.sym('v')  # [V]

        # Params
        self.g = sys_params['g']  # [cm/s^2]
        self.a = sys_params['a']  # [cm^2]
        self.A = sys_params['A']  # [cm^2]
        self.k = sys_params['k']  # [cm^3/Vs]

        self.ode = (self.k / self.A) * self.v - (self.a / self.A) * cs.sqrt(2 * self.g * self.h)
        self.dae = {'x': self.h, 'p': self.v, 't': self.t, 'ode': self.ode}

    def run(self, np_t, np_v, np_ic, output_t0=True):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': output_t0})
        sol = integrator(x0=np.reshape(np_ic, self.h.shape),
                         p=np.reshape(np_v, self.v.shape))
        return np.transpose(np.array(sol['xf']))

    def get_runge_kutta(self, T, runge_kutta_steps=4):
        DT = T / runge_kutta_steps
        f = cs.Function('f', [self.v, self.h], [self.ode])

        y0 = cs.MX.sym('y0')

        u = cs.MX.sym('u')
        y = y0

        for j in range(runge_kutta_steps):
            k1 = f(u, y)
            k2 = f(u, y + DT / 2 * k1)
            k3 = f(u, y + DT / 2 * k2)
            k4 = f(u, y + DT * k3)
            y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [u, y0], [y], ['u', 'y0'], ['yf'])
