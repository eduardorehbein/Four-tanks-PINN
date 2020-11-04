import numpy as np
from util.plot import Plotter


g = 981.0  # [cm/s^2]
a1 = 0.071  # [cm^2]
a2 = 0.057  # [cm^2]
a3 = 0.071  # [cm^2]
a4 = 0.057  # [cm^2]
A1 = 28.0  # [cm^2]
A2 = 32.0  # [cm^2]
A3 = 28.0  # [cm^2]
A4 = 32.0  # [cm^2]
alpha1 = 0.5  # [adm]
alpha2 = 0.5  # [adm]
k1 = 3.33  # [cm^3/Vs]
k2 = 3.35  # [cm^3/Vs]

lowest_v = 0.5
highest_v = 3.38

var_v = np.linspace(lowest_v, highest_v, 50)
base_v = np.linspace(lowest_v, highest_v, 5)

Hs = {'var_v1': dict(), 'var_v2': dict()}

for v2 in base_v:
    h4 = (1 / (2 * g)) * (((1 - alpha1) * k1 * var_v) / a4) ** 2
    h3 = (1 / (2 * g)) * (((1 - alpha2) * k2 * v2) / a3) ** 2
    h2 = (1 / (2 * g * (a2 ** 2))) * (a4 * np.sqrt(2 * g * h4) + alpha2 * k2 * v2) ** 2
    h1 = (1 / (2 * g * (a1 ** 2))) * (a3 * np.sqrt(2 * g * h3) + alpha1 * k1 * var_v) ** 2

    Hs['var_v1'][v2] = np.array([h1, h2, h3, h4])

for v1 in base_v:
    h4 = (1 / (2 * g)) * (((1 - alpha1) * k1 * v1) / a4) ** 2
    h3 = (1 / (2 * g)) * (((1 - alpha2) * k2 * var_v) / a3) ** 2
    h2 = (1 / (2 * g * (a2 ** 2))) * (a4 * np.sqrt(2 * g * h4) + alpha2 * k2 * var_v) ** 2
    h1 = (1 / (2 * g * (a1 ** 2))) * (a3 * np.sqrt(2 * g * h3) + alpha1 * k1 * v1) ** 2

    Hs['var_v2'][v1] = np.array([h1, h2, h3, h4])

plotter = Plotter()

# Plots with v2 constant
plotter.plot(var_v,
             [Hs['var_v1'][v2][0] for v2 in Hs['var_v1'].keys()],
             ['$v_2$ = ' + str(v2) for v2 in Hs['var_v1'].keys()],
             '$h_1$ with $v_2$ constant',
             '$v_1$',
             '$h_1$')
plotter.plot(var_v,
             [Hs['var_v1'][v2][1] for v2 in Hs['var_v1'].keys()],
             ['$v_2$ = ' + str(v2) for v2 in Hs['var_v1'].keys()],
             '$h_2$ with $v_2$ constant',
             '$v_1$',
             '$h_2$')
plotter.plot(var_v,
             [np.tile(Hs['var_v1'][v2][2], (var_v.size,)) for v2 in Hs['var_v1'].keys()],
             ['$v_2$ = ' + str(v2) for v2 in Hs['var_v1'].keys()],
             '$h_3$ with $v_2$ constant',
             '$v_1$',
             '$h_3$')
plotter.plot(var_v,
             [Hs['var_v1'][v2][3] for v2 in Hs['var_v1'].keys()],
             ['$v_2$ = ' + str(v2) for v2 in Hs['var_v1'].keys()],
             '$h_4$ with $v_2$ constant',
             '$v_1$',
             '$h_4$')

# Plots with v1 constant
plotter.plot(var_v,
             [Hs['var_v2'][v1][0] for v1 in Hs['var_v2'].keys()],
             ['$v_1$ = ' + str(v1) for v1 in Hs['var_v2'].keys()],
             '$h_1$ with $v_1$ constant',
             '$v_2$',
             '$h_1$')
plotter.plot(var_v,
             [Hs['var_v2'][v1][1] for v1 in Hs['var_v2'].keys()],
             ['$v_1$ = ' + str(v1) for v1 in Hs['var_v2'].keys()],
             '$h_2$ with $v_1$ constant',
             '$v_2$',
             '$h_2$')
plotter.plot(var_v,
             [Hs['var_v2'][v1][2] for v1 in Hs['var_v2'].keys()],
             ['$v_1$ = ' + str(v1) for v1 in Hs['var_v2'].keys()],
             '$h_3$ with $v_1$ constant',
             '$v_2$',
             '$h_3$')
plotter.plot(var_v,
             [np.tile(Hs['var_v2'][v1][3], (var_v.size,)) for v1 in Hs['var_v2'].keys()],
             ['$v_1$ = ' + str(v1) for v1 in Hs['var_v2'].keys()],
             '$h_4$ with $v_1$ constant',
             '$v_2$',
             '$h_4$')

plotter.show()

v1 = 3.38
v2 = 3.38

h4 = (1 / (2 * g)) * (((1 - alpha1) * k1 * v1) / a4) ** 2
h3 = (1 / (2 * g)) * (((1 - alpha2) * k2 * v2) / a3) ** 2
h2 = (1 / (2 * g * (a2 ** 2))) * (a4 * np.sqrt(2 * g * h4) + alpha2 * k2 * v2) ** 2
h1 = (1 / (2 * g * (a1 ** 2))) * (a3 * np.sqrt(2 * g * h3) + alpha1 * k1 * v1) ** 2

print('Operation point example: v = (' + str(v1) + ', ' + str(v2) +
      ') h = (' + str(h1) + ', ' + str(h2) + ', ' + str(h3) + ', ' + str(h4) + ')')
