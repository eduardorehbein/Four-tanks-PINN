import numpy as np
import casadi as cs


class CasadiSimulator:
    def __init__(self, sys_params):
        # Time
        self.t = cs.MX.sym('t')  # [s]

        # States
        self.h1 = cs.MX.sym('h1')  # [cm]
        self.h2 = cs.MX.sym('h2')  # [cm]
        self.h3 = cs.MX.sym('h3')  # [cm]
        self.h4 = cs.MX.sym('h4')  # [cm]

        self.states = cs.horzcat(self.h1, self.h2, self.h3, self.h4)

        # Controls
        self.v1 = cs.MX.sym('v1')  # [V]
        self.v2 = cs.MX.sym('v2')  # [V]

        self.controls = cs.horzcat(self.v1, self.v2)

        # Params
        self.g = sys_params['g']  # [cm/s^2]

        self.a1 = sys_params['a1']  # [cm^2]
        self.a2 = sys_params['a2']  # [cm^2]
        self.a3 = sys_params['a3']  # [cm^2]
        self.a4 = sys_params['a4']  # [cm^2]

        self.A1 = sys_params['A1']  # [cm^2]
        self.A2 = sys_params['A2']  # [cm^2]
        self.A3 = sys_params['A3']  # [cm^2]
        self.A4 = sys_params['A4']  # [cm^2]

        self.alpha1 = sys_params['alpha1']  # [adm]
        self.alpha2 = sys_params['alpha2']  # [adm]

        self.k1 = sys_params['k1']  # [cm^3/Vs]
        self.k2 = sys_params['k2']  # [cm^3/Vs]

        self.ode = cs.horzcat(-(self.a1 / self.A1) * cs.sqrt(2 * self.g * self.h1) +
                              (self.a3 / self.A1) * cs.sqrt(2 * self.g * self.h3) +
                              ((self.alpha1 * self.k1) / self.A1) * self.v1,
                              -(self.a2 / self.A2) * cs.sqrt(2 * self.g * self.h2) +
                              (self.a4 / self.A2) * cs.sqrt(2 * self.g * self.h4) +
                              ((self.alpha2 * self.k2) / self.A2) * self.v2,
                              -(self.a3 / self.A3) * cs.sqrt(2 * self.g * self.h3) +
                              (((1 - self.alpha2) * self.k2) / self.A3) * self.v2,
                              -(self.a4 / self.A4) * cs.sqrt(2 * self.g * self.h4) +
                              (((1 - self.alpha1) * self.k1) / self.A4) * self.v1)
        self.dae = {'x': cs.transpose(self.states), 'p': cs.transpose(self.controls), 't': self.t, 'ode': cs.transpose(self.ode)}

    def run(self, np_t, np_v, np_ic, output_t0=True):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': output_t0})
        sol = integrator(x0=np.reshape(np_ic, self.states.shape),
                         p=np.reshape(np_v, self.controls.shape))
        return np.transpose(np.array(sol['xf']))
