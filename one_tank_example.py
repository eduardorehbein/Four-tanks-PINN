import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import OneTankPINN
from util.plot import Plotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Instance PINN
model = OneTankPINN(sys_params=sys_params, hidden_layers=2, units_per_layer=10)
model.trained_T = 10.0

# Load model
model.load('models/one_tank/2020-08-06-11-24-21-10s-2l-10n-best-model')

# Test data
test_df = pd.read_csv('data/one_tank/rand_seed_10_sim_time_160.0s_1600_collocation_points.csv')

np_test_t = test_df['t'].to_numpy()
np_test_v = test_df['v'].to_numpy()
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_y = test_df[['h']].to_numpy()
np_test_ic = np.reshape(np_test_y[0, :], (1, np_test_y.shape[1]))

# Model prediction
test_T = 10.0
np_test_nn = model.predict(np_test_X, np_test_ic, prediction_T=test_T)

# Plot test results
plotter = Plotter()

markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
mse = (np.square(np_test_nn - np_test_y)).mean()
plotter.plot(x_axis=np_test_t,
             y_axis_list=[np_test_v, np_test_y, np_test_nn],
             labels=['$v$', '$\\hat{y}$', '$y$'],
             title='One tank model test. MSE: ' + str(round(mse, 3)),
             x_label='Time',
             y_label='Inputs and outputs',
             line_styles=['-', '--', 'o-'],
             markevery=markevery)
plotter.show()
