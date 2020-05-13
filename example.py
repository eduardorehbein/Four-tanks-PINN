import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, integrator, vertcat, sqrt

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
np_t = np.linspace(0, 50, 1000)
integrator = integrator('integrator', 'cvodes', dae, {'grid': np_t, 'output_t0': True})
sol = integrator(x0=[0, 0, 0, 0], p=[981,  # g [cm/s^2]
                                     0.071,  # a1 [cm^2]
                                     0.057,  # a2 [cm^2]
                                     0.071,  # a3 [cm^2]
                                     0.057,  # a4 [cm^2]
                                     28,  # A1 [cm^2]
                                     32,  # A2 [cm^2]
                                     28,  # A3 [cm^2]
                                     32,  # A4 [cm^2]
                                     0.5,  # alpha1 [adm]
                                     0.5,  # alpha2 [adm]
                                     3.33,  # k1 [cm^3/Vs]
                                     3.35,  # k2 [cm^3/Vs]
                                     3.0,  # v1 [V]
                                     3.0  # v2 [V]
                                     ])

np_h = np.array(sol['xf'])
plt.plot(np_t, np_h[0], label='h1')
plt.plot(np_t, np_h[1], label='h2')
plt.plot(np_t, np_h[2], label='h3')
plt.plot(np_t, np_h[3], label='h4')
plt.legend()
plt.show()
