import casadi as cs
import numpy as np


class CasadiSimulator:
    def __init__(self, sys_params):
        # Parameters dictionary
        self.sys_params = sys_params

        # Time
        self.t = cs.SX.sym('t')

        # State
        self.h = cs.SX.sym('h')

        # Control
        self.v = cs.SX.sym('v')

        # Params
        self.g = cs.SX.sym('g')
        self.a = cs.SX.sym('a')
        self.A = cs.SX.sym('A')
        self.k = cs.SX.sym('k')

        self.params = cs.vertcat(self.g, self.a, self.A, self.k, self.v)
        self.rhs = cs.vertcat((self.k / self.A) * self.v - (self.a / self.A) * cs.sqrt(2 * self.g * self.h))
        self.dae = {'x': self.h, 'p': self.params, 't': self.t, 'ode': self.rhs}

    def run(self, np_t, np_v, np_ic):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t[0], 'output_t0': True})
        sol = integrator(x0=np_ic, p=[self.sys_params['g'],  # g [cm/s^2]
                                      self.sys_params['a'],  # a [cm^2]
                                      self.sys_params['A'],  # A [cm^2]
                                      self.sys_params['k'],  # k [cm^3/Vs]
                                      np_v,  # v [V]
                                      ])
        return np.array(sol['xf'])
