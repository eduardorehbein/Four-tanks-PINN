import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, integrator, vertcat, sqrt

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a1': 0.071,  # [cm^2]
              'a2': 0.057,  # [cm^2]
              'a3': 0.071,  # [cm^2]
              'a4': 0.057,  # [cm^2]
              'A1': 28.0,  # [cm^2]
              'A2': 32.0,  # [cm^2]
              'A3': 28.0,  # [cm^2]
              'A4': 32.0,  # [cm^2]
              'alpha1': 0.5,  # [adm]
              'alpha2': 0.5,  # [adm]
              'k1': 3.33,  # [cm^3/Vs]
              'k2': 3.35,  # [cm^3/Vs]
              }

# Control values
v = [3.0, 3.0]

# Time analysis
T1 = (sys_params['A1'] / (sys_params['g'] * (sys_params['a1'] ** 2))) * \
     ((1 - sys_params['alpha2']) * sys_params['k2'] * v[1] + sys_params['alpha1'] * sys_params['k1'] * v[0])
T2 = (sys_params['A2'] / (sys_params['g'] * (sys_params['a2'] ** 2))) * \
     ((1 - sys_params['alpha1']) * sys_params['k1'] * v[0] + sys_params['alpha2'] * sys_params['k2'] * v[1])
T3 = (sys_params['A3'] / (sys_params['g'] * (sys_params['a3'] ** 2))) * \
     ((1 - sys_params['alpha2']) * sys_params['k2'] * v[1])
T4 = (sys_params['A4'] / (sys_params['g'] * (sys_params['a4'] ** 2))) * \
     ((1 - sys_params['alpha1']) * sys_params['k1'] * v[0])
print('Time constant T1: ' + str(T1))
print('Time constant T2: ' + str(T2))
print('Time constant T3: ' + str(T3))
print('Time constant T4: ' + str(T4))

s_r_t = min(T1, T2, T3, T4)
print('The shortest response time (SRT) is ' + str(s_r_t))

ol_sample_T = (3 / 20) * s_r_t
print('System\'s open loop sample time is going to be: ' + str(ol_sample_T) + ' seconds. Obs: sample time = 3*SRT/20')

# Casadi simulation
# Time
t = SX.sym('t')

# States
h1 = SX.sym('h1')
h2 = SX.sym('h2')
h3 = SX.sym('h3')
h4 = SX.sym('h4')

states = vertcat(h1, h2, h3, h4)

# Controls
v1 = SX.sym('v1')
v2 = SX.sym('v2')

# Params
g = SX.sym('g')

a1 = SX.sym('a1')
a2 = SX.sym('a2')
a3 = SX.sym('a3')
a4 = SX.sym('a4')

A1 = SX.sym('A1')
A2 = SX.sym('A2')
A3 = SX.sym('A3')
A4 = SX.sym('A4')

alpha1 = SX.sym('alpha1')
alpha2 = SX.sym('alpha2')

k1 = SX.sym('k1')
k2 = SX.sym('k2')

params = vertcat(g, a1, a2, a3, a4, A1, A2, A3, A4, alpha1, alpha2, k1, k2, v1, v2)
rhs = vertcat(-(a1/A1)*sqrt(2*g*h1) + (a3/A1)*sqrt(2*g*h3) + ((alpha1*k1)/A1)*v1,
              -(a2/A2)*sqrt(2*g*h2) + (a4/A2)*sqrt(2*g*h4) + ((alpha2*k2)/A2)*v2,
              -(a3/A3)*sqrt(2*g*h3) + (((1 - alpha2)*k2)/A3)*v2,
              -(a4/A4)*sqrt(2*g*h4) + (((1 - alpha1)*k1)/A4)*v1)
dae = {'x': states, 'p': params, 't': t, 'ode': rhs}
np_t = np.linspace(0, ol_sample_T, 100)
integrator = integrator('integrator', 'cvodes', dae, {'grid': np_t, 'output_t0': True})

sol = integrator(x0=[0, 0, 0, 0], p=[sys_params['g'],  # g [cm/s^2]
                                     sys_params['a1'],  # a1 [cm^2]
                                     sys_params['a2'],  # a2 [cm^2]
                                     sys_params['a3'],  # a3 [cm^2]
                                     sys_params['a4'],  # a4 [cm^2]
                                     sys_params['A1'],  # A1 [cm^2]
                                     sys_params['A2'],  # A2 [cm^2]
                                     sys_params['A3'],  # A3 [cm^2]
                                     sys_params['A4'],  # A4 [cm^2]
                                     sys_params['alpha1'],  # alpha1 [adm]
                                     sys_params['alpha2'],  # alpha2 [adm]
                                     sys_params['k1'],  # k1 [cm^3/Vs]
                                     sys_params['k2'],  # k2 [cm^3/Vs]
                                     v[0],  # v1 [V]
                                     v[1]  # v2 [V]
                                     ])

np_h = np.array(sol['xf'])
plt.plot(np_t, np_h[0], label='h1')
plt.plot(np_t, np_h[1], label='h2')
plt.plot(np_t, np_h[2], label='h3')
plt.plot(np_t, np_h[3], label='h4')
plt.legend()
plt.show()
