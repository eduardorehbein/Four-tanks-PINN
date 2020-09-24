import numpy as np
import tensorflow as tf
import pandas as pd
from util.tests import TTester, TTestContainer
from util.pinn import VanDerPolPINN

# Working period test's parameters
train_Ts_to_test = (0.01, 0.1, 1.0, 2.0, 4.0, 8.0)

# Neural network's parameters
hidden_layers = 10
units_per_layer = 5

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 2000
val_T = 0.5
test_T = val_T

# Other parameters
random_seed = 30

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'van_der_pol'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Load data into a container
data_container = TTestContainer()

# Validation data
val_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_60_sim_time_10.0s_10_scenarios_200_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'u']].to_numpy()
data_container.np_val_Y = val_df[['x1', 'x2']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['x1', 'x2']].to_numpy()
data_container.val_T = val_T

# Test data
test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_10_sim_time_10.0s_200_collocation_points.csv')
data_container.np_test_t = test_df['t'].to_numpy()
data_container.np_test_X = test_df[['t', 'u']].to_numpy()
np_test_Y = test_df[['x1', 'x2']].to_numpy()
data_container.np_test_Y = np_test_Y
data_container.np_test_ic = np.reshape(np_test_Y[0], (1, np_test_Y.shape[1]))
data_container.test_T = test_T

for working_period in train_Ts_to_test:
    train_df = pd.read_csv('data/van_der_pol/rand_seed_30_T_' + str(working_period) +
                           's_1000_scenarios_100_collocation_points.csv')

    # Train data
    train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
    np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
    np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
    np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

    data_container.set_train_u_X(working_period, np_train_u_X)
    data_container.set_train_u_Y(working_period, np_train_u_Y)
    data_container.set_train_f_X(working_period, np_train_f_X)

# Test
tester = TTester(VanDerPolPINN, hidden_layers, units_per_layer, train_Ts_to_test, adam_epochs, max_lbfgs_iterations)
tester.test(data_container, results_subdirectory, save_mode='all')
