import casadi as cs
import numpy as np


class CasadiSimulator:
    def __init__(self):
        # Time
        self.t = cs.SX.sym('t')

        # States
        self.x1 = cs.SX.sym('x1')
        self.x2 = cs.SX.sym('x2')

        self.states = cs.vertcat(self.x1, self.x2)

        # Control
        self.u = cs.SX.sym('u')

        self.ode = cs.vertcat((1 - self.x2 ** 2) * self.x1 - self.x2 + self.u,
                              self.x1)
        self.dae = {'x': self.states, 'p': self.u, 't': self.t, 'ode': self.ode}

    def run(self, np_t, np_u, np_ic):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': True})
        sol = integrator(x0=np_ic, p=np_u)
        return np.transpose(np.array(sol['xf']))
