import casadi as cs
import numpy as np


class OneTankSystem:
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


class VanDerPolSystem:
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


class FourTanksSystem:
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

    def get_runge_kutta(self, T, runge_kutta_steps=4):
        DT = T / runge_kutta_steps
        f = cs.Function('f', [self.controls, self.states], [self.ode])

        y0 = cs.MX.sym('y0', 1, 4)

        v1 = cs.MX.sym('v1')
        v2 = cs.MX.sym('v2')
        u = cs.horzcat(v1, v2)
        y = y0

        for j in range(runge_kutta_steps):
            k1 = f(u, y)
            k2 = f(u, y + DT / 2 * k1)
            k3 = f(u, y + DT / 2 * k2)
            k4 = f(u, y + DT * k3)
            y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [u, y0], [y], ['u', 'y0'], ['yf'])