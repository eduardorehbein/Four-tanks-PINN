import numpy as np
import casadi as cs


class ResponseAnalyser:
    def __init__(self, sys_params):
        self.sys_params = sys_params

    def get_time_constants(self, np_v):
        T1 = (self.sys_params['A1'] / (self.sys_params['g'] * (self.sys_params['a1'] ** 2))) * \
             ((1 - self.sys_params['alpha2']) * self.sys_params['k2'] * np_v[1] +
              self.sys_params['alpha1'] * self.sys_params['k1'] * np_v[0])
        T2 = (self.sys_params['A2'] / (self.sys_params['g'] * (self.sys_params['a2'] ** 2))) * \
             ((1 - self.sys_params['alpha1']) * self.sys_params['k1'] * np_v[0] +
              self.sys_params['alpha2'] * self.sys_params['k2'] * np_v[1])
        T3 = (self.sys_params['A3'] / (self.sys_params['g'] * (self.sys_params['a3'] ** 2))) * \
             ((1 - self.sys_params['alpha2']) * self.sys_params['k2'] * np_v[1])
        T4 = (self.sys_params['A4'] / (self.sys_params['g'] * (self.sys_params['a4'] ** 2))) * \
             ((1 - self.sys_params['alpha1']) * self.sys_params['k1'] * np_v[0])

        return T1, T2, T3, T4

    def get_ol_sample_time(self, np_vs):
        # Response and sample time analysis
        lowest_T1 = float('inf')
        lowest_T2 = float('inf')
        lowest_T3 = float('inf')
        lowest_T4 = float('inf')

        for i in range(np_vs.shape[1]):
            np_v = np_vs[:, i]
            T1, T2, T3, T4 = self.get_time_constants(np_v)
            lowest_T1 = min(lowest_T1, T1)
            lowest_T2 = min(lowest_T2, T2)
            lowest_T3 = min(lowest_T3, T3)
            lowest_T4 = min(lowest_T4, T4)

        print('Lowest time constant T1: ' + str(lowest_T1))
        print('Lowest time constant T2: ' + str(lowest_T2))
        print('Lowest time constant T3: ' + str(lowest_T3))
        print('Lowest time constant T4: ' + str(lowest_T4))

        f_r_t = min(lowest_T1, lowest_T2, lowest_T3, lowest_T4)
        print('The fastest response time (FRT) is ' + str(f_r_t))

        ol_sample_time = (3 / 20) * f_r_t
        print('System\'s open loop sample time is: ' + str(
            ol_sample_time) + ' seconds. Obs: sample time = 3*FRT/20')

        return ol_sample_time


class CasadiSimulator:
    def __init__(self, sys_params):
        # Time
        self.t = cs.SX.sym('t')  # [s]

        # States
        self.h1 = cs.SX.sym('h1')  # [cm]
        self.h2 = cs.SX.sym('h2')  # [cm]
        self.h3 = cs.SX.sym('h3')  # [cm]
        self.h4 = cs.SX.sym('h4')  # [cm]

        self.states = cs.vertcat(self.h1, self.h2, self.h3, self.h4)

        # Controls
        self.v1 = cs.SX.sym('v1')  # [V]
        self.v2 = cs.SX.sym('v2')  # [V]

        self.controls = cs.vertcat(self.v1, self.v2)

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

        self.rhs = cs.vertcat(-(self.a1 / self.A1) * cs.sqrt(2 * self.g * self.h1) +
                              (self.a3 / self.A1) * cs.sqrt(2 * self.g * self.h3) +
                              ((self.alpha1 * self.k1) / self.A1) * self.v1,
                              -(self.a2 / self.A2) * cs.sqrt(2 * self.g * self.h2) +
                              (self.a4 / self.A2) * cs.sqrt(2 * self.g * self.h4) +
                              ((self.alpha2 * self.k2) / self.A2) * self.v2,
                              -(self.a3 / self.A3) * cs.sqrt(2 * self.g * self.h3) +
                              (((1 - self.alpha2) * self.k2) / self.A3) * self.v2,
                              -(self.a4 / self.A4) * cs.sqrt(2 * self.g * self.h4) +
                              (((1 - self.alpha1) * self.k1) / self.A4) * self.v1)
        self.dae = {'x': self.states, 'p': self.controls, 't': self.t, 'ode': self.rhs}

    def run(self, np_t, np_v, np_ic):
        integrator = cs.integrator('integrator', 'cvodes', self.dae, {'grid': np_t, 'output_t0': True})
        sol = integrator(x0=np_ic, p=np_v)
        return np.array(sol['xf'])
