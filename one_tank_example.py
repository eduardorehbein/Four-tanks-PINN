import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from util.pinn import OneTankPINN
from util.plot import PdfPlotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Instance PINN
model = OneTankPINN(sys_params=sys_params,
                    hidden_layers=2,
                    units_per_layer=10)

# Load model
model.load('models/one_tank/2020-08-06-11-24-21-10s-2l-10n-best-model')

# Test data
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_sim_time_160.0s_1600_collocation_points.csv')

np_test_t = test_df['t'].to_numpy()
np_test_v = test_df['v'].to_numpy()
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_y = test_df[['h']].to_numpy()
np_test_ic = np_test_y[0]

# Model prediction
np_test_nn = model.predict(np_test_X, np_test_ic, working_period=10.0)

# Plotter
plotter = PdfPlotter()

# Plot test results
plt.plot(np_test_t, np_test_v, label='$v$')
plt.plot(np_test_t, np_test_y.flatten(), label='$\hat{y}$')
plt.plot(np_test_t, np_test_nn.flatten(), label='$y$')
plt.legend()
plt.show()
