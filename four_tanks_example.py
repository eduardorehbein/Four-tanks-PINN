import numpy as np
import tensorflow as tf
import pandas as pd
from util.normalizer import Normalizer
from util.pinn import FourTanksPINN
from util.plot import Plotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

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

# Train data
train_df = pd.read_csv('data/four_tanks/rand_seed_30_T_15.0s_1000_scenarios_100_collocation_points.csv')

train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'v1', 'v2', 'h1_0', 'h2_0', 'h3_0', 'h4_0']].to_numpy()
np_train_u_Y = train_u_df[['h1', 'h2', 'h3', 'h4']].to_numpy()
np_train_f_X = train_df[['t', 'v1', 'v2', 'h1_0', 'h2_0', 'h3_0', 'h4_0']].sample(frac=1).to_numpy()

# Normalizers
X_normalizer = Normalizer()
Y_normalizer = Normalizer()

X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
Y_normalizer.parametrize(np_train_u_Y)

# Instance PINN
model = FourTanksPINN(sys_params=sys_params,
                      hidden_layers=5,
                      units_per_layer=15,
                      X_normalizer=X_normalizer,
                      Y_normalizer=Y_normalizer)

# Load model
model.load_weights('models/four_tanks/2020-06-08-22-25-06-15s-5l-15n-best-model.h5')
model.trained_T = 15.0

# Test data
test_df = pd.read_csv('data/four_tanks/rand_seed_10_sim_time_350.0s_350_collocation_points.csv')

np_test_t = test_df['t'].to_numpy()
np_test_v = test_df[['v1', 'v2']].to_numpy()
np_test_X = test_df[['t', 'v1', 'v2']].to_numpy()
np_test_Y = test_df[['h1', 'h2', 'h3', 'h4']].to_numpy()
np_test_ic = np.reshape(np_test_Y[0, :], (1, np_test_Y.shape[1]))

# Model prediction
test_T = 10.0
np_test_NN = model.predict(np_test_X, np_test_ic, prediction_T=test_T)

# Plot test results
plotter = Plotter()

markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
mse = (np.square(np_test_NN - np_test_Y)).mean()
plotter.plot(x_axis=np_test_t,
             y_axis_list=[np_test_v[:, 0], np_test_v[:, 1],
                          np_test_Y[:, 0], np_test_NN[:, 0],
                          np_test_Y[:, 1], np_test_NN[:, 1],
                          np_test_Y[:, 2], np_test_NN[:, 2],
                          np_test_Y[:, 3], np_test_NN[:, 3]],
             labels=['$v_1$', '$v_2$',
                     '$\\hat{y}_1$', '$y_1$',
                     '$\\hat{y}_2$', '$y_2$',
                     '$\\hat{y}_3$', '$y_3$',
                     '$\\hat{y}_4$', '$y_4$'],
             title='Four tanks model test. MSE: ' + str(round(mse, 3)),
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
