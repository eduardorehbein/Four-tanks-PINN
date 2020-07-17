import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import OneTankPINN
from util.tests import StructTester

# Structural parameters
layers_to_test = (1, 2, 3)
neurons_per_layer_to_test = (2, 3, 5, 8, 10, 15, 20)

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 1000

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'one_tank'

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

# Load data
df = pd.read_csv('data/one_tank/rand_seed_30_t_range_15.0s_2000_scenarios_100_collocation_points.csv')

# Train data
train_df = df[df['scenario'] <= 1000]
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'v', 'ic']].to_numpy()
np_train_u_Y = train_u_df[['h']].to_numpy()
np_train_f_X = train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

# Validation data
val_df = df[(df['scenario'] > 1000) & (df['scenario'] <= 1100)].sample(frac=1)
np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
np_val_Y = val_df[['h']].to_numpy()

# Test
tester = StructTester(layers_to_test, neurons_per_layer_to_test, adam_epochs, max_lbfgs_iterations)
tester.test(OneTankPINN, sys_params, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y, results_subdirectory)
