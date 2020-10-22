import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import FourTanksPINN
from util.tests import ExhaustionTester
from util.data_container import ExhaustionTestContainer


# Neural networks' parameters
hidden_layers = 5
units_per_layer = 20

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 10000
train_T = 15.0
val_T = 10.0

# Test parameters
test_T = val_T

# Other parameters
random_seed = 30

# Directory under 'results' and 'models' where the plots and models are going to be saved
results_and_models_subdirectory = 'four_tanks'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

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

# Load data into a container
data_container = ExhaustionTestContainer()

# Train data
train_df = pd.read_csv('data/four_tanks/rand_seed_30_T_' + str(train_T) + 's_1000_scenarios_100_collocation_points.csv')
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
data_container.np_train_u_X = train_u_df[['t', 'v1', 'v2', 'h1_0', 'h2_0', 'h3_0', 'h4_0']].to_numpy()
data_container.np_train_u_Y = train_u_df[['h1', 'h2', 'h3', 'h4']].to_numpy()
data_container.np_train_f_X = train_df[['t', 'v1', 'v2', 'h1_0', 'h2_0', 'h3_0', 'h4_0']].sample(frac=1).to_numpy()
data_container.train_T = train_T

# Validation data
val_df = pd.read_csv('data/four_tanks/long_signal_rand_seed_60_sim_time_350.0s_10_scenarios_350_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'v1', 'v2']].to_numpy()
data_container.np_val_Y = val_df[['h1', 'h2', 'h3', 'h4']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['h1', 'h2', 'h3', 'h4']].to_numpy()
data_container.val_T = val_T

# Test data
test_df = pd.read_csv('data/four_tanks/long_signal_rand_seed_10_sim_time_350.0s_350_collocation_points.csv')
data_container.np_test_t = test_df['t'].to_numpy()
data_container.np_test_X = test_df[['t', 'v1', 'v2']].to_numpy()
data_container.np_test_Y = test_df[['h1', 'h2', 'h3', 'h4']].to_numpy()
data_container.np_test_ic = np.reshape(data_container.np_test_Y[0], (1, data_container.np_test_Y.shape[1]))
data_container.test_T = test_T

# Tester
tester = ExhaustionTester(FourTanksPINN, hidden_layers, units_per_layer, adam_epochs, max_lbfgs_iterations, sys_params)

# Test
tester.test(data_container, results_and_models_subdirectory)
