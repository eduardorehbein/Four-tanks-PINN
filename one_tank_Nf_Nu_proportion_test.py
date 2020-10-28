import numpy as np
import tensorflow as tf
import pandas as pd
import time
from util.pinn import OneTankPINN
from util.tests import NfNuTester
from util.data_container import NfNuTestContainer
from util.data_interface import TrainDataGenerator

# Nf/Nu proportion test's parameters
nfs_to_test = (2000, 4000, 10000, 100000)
nus_to_test = (40, 80, 100, 500, 1000)

# Neural network's parameters
hidden_layers = 2
units_per_layer = 10

# Train data parameters
scenarios = 1000
collocation_points = 100

np_lowest_v = np.array([0.5])
np_highest_v = np.array([4.45])
np_lowest_h = np.array([2.0])
np_highest_h = np.array([20.0])

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 2000
train_T = 10.0
val_T = train_T

# Other parameters
random_seed = int(time.time())

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'one_tank'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Load data into a container
data_container = NfNuTestContainer()
data_container.random_seed = random_seed
data_container.train_T = train_T

# Validation data
val_df = pd.read_csv('data/one_tank/rand_seed_60_sim_time_160.0s_1600_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'v']].to_numpy()
data_container.np_val_Y = val_df[['h']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['h']].to_numpy()
data_container.val_T = val_T

# Train data generator
train_data_gen = TrainDataGenerator(np_lowest_u=np_lowest_v,
                                    np_highest_u=np_highest_v,
                                    np_lowest_y=np_lowest_h,
                                    np_highest_y=np_highest_h)

for nf in nfs_to_test:
    for nu in nus_to_test:
        # Train data
        np_train_u_X, np_train_u_Y, np_train_f_X = train_data_gen.get_data(nu, int(nf/nu), train_T)
        data_container.set_train_u_X(nf, nu, np_train_u_X)
        data_container.set_train_u_Y(nf, nu, np_train_u_Y)
        data_container.set_train_f_X(nf, nu, np_train_f_X)

# Test
tester = NfNuTester(OneTankPINN, hidden_layers, units_per_layer, nfs_to_test, nus_to_test,
                    adam_epochs, max_lbfgs_iterations, sys_params)
tester.test(data_container, results_subdirectory)
