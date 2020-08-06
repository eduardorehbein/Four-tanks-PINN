import casadi as cs
import numpy as np


class CasadiSimulator:
    def __init__(self, sys_params):
        # Time
        self.t = cs.SX.sym('t')  # [s]

        # State
        self.h = cs.SX.sym('h')  # [cm]

        # Control
        self.v = cs.SX.sym('v')  # [V]

        # Params
        self.g = sys_params['g']  # [cm/s^2]
        self.a = sys_params['a']  # [cm^2]
        self.A = sys_params['A']  # [cm^2]
        self.k = sys_params['k']  # [cm^3/Vs]

        self.ode = (self.k / self.A) * self.v - (self.a / self.A) * cs.sqrt(2 * self.g * self.h)
        self.dae = {'x': self.h, 'p': self.v, 't': self.t, 'ode': self.ode}

    def run(self, np_t, np_v, np_ic):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': True})
        sol = integrator(x0=np_ic, p=np_v)
        return np.array(sol['xf'])
