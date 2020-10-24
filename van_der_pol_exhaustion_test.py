import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.tests import ExhaustionTester
from util.data_container import ExhaustionTestContainer
from util.data_interface import TrainDataGenerator


# Neural networks' parameters
hidden_layers = 4
units_per_layer = 15

# Train data parameters
scenarios = 1000
collocation_points = 10

np_lowest_u = np.array([-1.0])
np_highest_u = np.array([1.0])
np_lowest_x = np.array([-3.0, -3.0])
np_highest_x = np.array([3.0, 3.0])

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 10000
train_T = 0.5
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

# Load data into a container
data_container = ExhaustionTestContainer()

# Train data
train_data_gen = TrainDataGenerator(np_lowest_u=np_lowest_u,
                                    np_highest_u=np_highest_u,
                                    np_lowest_y=np_lowest_x,
                                    np_highest_y=np_highest_x)
data_container.np_train_u_X, data_container.np_train_u_Y, data_container.np_train_f_X = \
    train_data_gen.get_data(scenarios, collocation_points, train_T)
data_container.train_T = train_T

# Validation data
val_df = pd.read_csv('data/van_der_pol/rand_seed_60_sim_time_10.0s_10_scenarios_200_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'u']].to_numpy()
data_container.np_val_Y = val_df[['x1', 'x2']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['x1', 'x2']].to_numpy()
data_container.val_T = val_T

# Test data
test_df = pd.read_csv('data/van_der_pol/rand_seed_10_sim_time_10.0s_200_collocation_points.csv')
data_container.np_test_t = test_df['t'].to_numpy()
data_container.np_test_X = test_df[['t', 'u']].to_numpy()
data_container.np_test_Y = test_df[['x1', 'x2']].to_numpy()
data_container.np_test_ic = np.reshape(data_container.np_test_Y[0, :], (1, data_container.np_test_Y.shape[1]))
data_container.test_T = test_T

# Tester
tester = ExhaustionTester(VanDerPolPINN, hidden_layers, units_per_layer, adam_epochs, max_lbfgs_iterations,
                          random_seed=random_seed)

# Test
tester.test(data_container, results_and_models_subdirectory)
