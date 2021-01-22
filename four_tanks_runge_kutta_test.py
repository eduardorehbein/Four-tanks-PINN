import numpy as np
import tensorflow as tf
import pandas as pd
from util.systems import FourTanksSystem
from util.plot import Plotter


# Parallel threads setup
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# System parameters dictionary
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

# Simulator instance
simulator = FourTanksSystem(sys_params)
test_T = 10.0
runge_kutta = simulator.get_runge_kutta(test_T)

# Test data
test_df = pd.read_csv('data/four_tanks/rand_seed_10_sim_time_350.0s_350_collocation_points.csv')
test_df = test_df[(test_df['t'] % test_T) == 0]

np_test_t = test_df['t'].to_numpy()
np_test_v = test_df[['v1', 'v2']].to_numpy()
np_test_Y = test_df[['h1', 'h2', 'h3', 'h4']].to_numpy()

# Model prediction
np_rk = np.reshape(np_test_Y[0, :], (1, np_test_Y.shape[1]))
for np_t, np_v, np_y0 in zip(np_test_t[:-1], np_test_v[:-1, :], np_test_Y[:-1, :]):
    cs_rk = runge_kutta(u=np_v, y0=np_y0)['yf']
    np_rk = np.append(np_rk, cs_rk, axis=0)

# Plot
plotter = Plotter()

markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
plotter.plot(x_axis=np_test_t,
             y_axis_list=[np_test_v[:, 0], np_test_v[:, 1],
                          np_test_Y[:, 0], np_rk[:, 0],
                          np_test_Y[:, 1], np_rk[:, 1],
                          np_test_Y[:, 2], np_rk[:, 2],
                          np_test_Y[:, 3], np_rk[:, 3]],
             labels=['$v_1$', '$v_2$',
                     '$\\hat{y}_1$', '$y_1$',
                     '$\\hat{y}_2$', '$y_2$',
                     '$\\hat{y}_3$', '$y_3$',
                     '$\\hat{y}_4$', '$y_4$'],
             title='Four tanks Runge-Kutta test',
             x_label='Time',
             y_label='Inputs and outputs',
             line_styles=['-', '-',
                          '--', 'o-',
                          '--', 'o-',
                          '--', 'o-',
                          '--', 'o-'],
             markevery=markevery,
             draw_styles=['steps', 'steps',
                          'default', 'default',
                          'default', 'default',
                          'default', 'default',
                          'default', 'default'])

plotter.show()
