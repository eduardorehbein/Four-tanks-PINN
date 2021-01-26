import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import OneTankPINN
from util.tests import ExhaustionTester
from util.data_container import ExhaustionTestContainer
from util.data_interface import TrainDataGenerator


# Neural networks parameters
hidden_layers = 2
units_per_layer = 10

# Training data parameters
scenarios = 1000
collocation_points = 100

np_lowest_v = np.array([0.5])
np_highest_v = np.array([4.45])
np_lowest_h = np.array([2.0])
np_highest_h = np.array([20.0])

# Training parameters
adam_epochs = 500
max_lbfgs_iterations = 20000
train_T = 10.0
val_T = train_T

# Test parameters
test_T = val_T

# Other parameters
random_seed = 30

# Directory under 'results' and 'models' where the plots and models will be saved
results_and_models_subdirectory = 'one_tank'

# Parallel threads setup
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# System parameters dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Loading data into a container
data_container = ExhaustionTestContainer()

# Training data
train_data_gen = TrainDataGenerator(np_lowest_u=np_lowest_v,
                                    np_highest_u=np_highest_v,
                                    np_lowest_y=np_lowest_v,
                                    np_highest_y=np_highest_v)
data_container.np_train_u_X, data_container.np_train_u_Y, data_container.np_train_f_X = \
    train_data_gen.get_data(scenarios, collocation_points, train_T)
data_container.train_T = train_T

# Validation data
val_df = pd.read_csv('data/one_tank/rand_seed_60_sim_time_160.0s_1600_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'v']].to_numpy()
data_container.np_val_Y = val_df[['h']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['h']].to_numpy()
data_container.val_T = val_T

# Test data
test_df = pd.read_csv('data/one_tank/rand_seed_10_sim_time_160.0s_1600_collocation_points.csv')
data_container.np_test_t = test_df['t'].to_numpy()
data_container.np_test_X = test_df[['t', 'v']].to_numpy()
np_test_Y = test_df[['h']].to_numpy()
data_container.np_test_Y = np_test_Y
data_container.np_test_ic = np.reshape(np_test_Y[0, :], (1, np_test_Y.shape[1]))
data_container.test_T = test_T

# Tester
tester = ExhaustionTester(OneTankPINN, hidden_layers, units_per_layer, adam_epochs, max_lbfgs_iterations,
                          sys_params, random_seed)

# Test
tester.test(data_container, results_and_models_subdirectory)
