import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.tests import ExhaustionTester


# Neural networks' parameters
hidden_layers = 4
units_per_layer = 20

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 10000
train_T = 1.0
val_T = 0.5

# Test parameters
test_T = val_T

# Other parameters
random_seed = 30

# Directory under 'results' and 'models' where the plots and models are going to be saved
results_and_models_subdirectory = 'van_der_pol'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Load train and validation data
train_df = pd.read_csv('data/van_der_pol/rand_seed_30_T_' + str(train_T) + 's_1000_scenarios_10_collocation_points.csv')

# Train data
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

# Validation data
val_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_60_sim_time_10.0s_10_scenarios_200_collocation_points.csv')
np_val_X = val_df[['t', 'u']].to_numpy()
np_val_Y = val_df[['x1', 'x2']].to_numpy()
np_val_ic = val_df[val_df['t'] == 0.0][['x1', 'x2']].to_numpy()

# Test data
test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_10_sim_time_10.0s_200_collocation_points.csv')
np_test_t = test_df['t'].to_numpy()
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_Y = test_df[['x1', 'x2']].to_numpy()
np_test_ic = np.reshape(np_test_Y[0], (1, np_test_Y.shape[1]))

# Tester
tester = ExhaustionTester(VanDerPolPINN, hidden_layers, units_per_layer, adam_epochs, max_lbfgs_iterations)

# Test
tester.test(np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
            np_test_t, np_test_X, np_test_ic, test_T, np_test_Y, results_and_models_subdirectory, save_mode='all')
