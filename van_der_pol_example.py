import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.plot import Plotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Instance PINN
model = VanDerPolPINN(hidden_layers=4, units_per_layer=20)
model.trained_T = 0.5

# Load model
model.load('models/van_der_pol/2020-10-25-04-19-43-0dot5s-4l-20n-exhausted-model')

# Test data
test_df = pd.read_csv('data/van_der_pol/rand_seed_10_sim_time_10.0s_200_collocation_points.csv')

np_test_t = test_df['t'].to_numpy()
np_test_u = test_df['u'].to_numpy()
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_y = test_df[['x1', 'x2']].to_numpy()
np_test_ic = np.reshape(np_test_y[0, :], (1, np_test_y.shape[1]))

# Model prediction
test_T = 0.5
np_test_nn = model.predict(np_test_X, np_test_ic, prediction_T=test_T)

# Plot test results
plotter = Plotter()

markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
mse = (np.square(np_test_nn - np_test_y)).mean()
plotter.plot(x_axis=np_test_t,
             y_axis_list=[np_test_u, np_test_y[:, 0], np_test_nn[:, 0], np_test_y[:, 1], np_test_nn[:, 1]],
             labels=['$u$', '$\\hat{y}_1$', '$y_1$', '$\\hat{y}_2$', '$y_2$'],
             title='Van der Pol model test',
             x_label='Time',
             y_label=None,
             line_styles=['-', '--', 'o-', '--', 'o-'],
             markevery=markevery,
             draw_styles=['steps', 'default', 'default', 'default', 'default'])
plotter.show()
