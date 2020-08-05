import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import OneTankPINN
from util.tests import BestAndWorstModelTestContainer, BestAndWorstModelTester


# Neural networks' parameters
best_model_hidden_layers = 2
best_model_units_per_layer = 10

worst_model_hidden_layers = 2
worst_model_units_per_layer = 10

# Train parameters
adam_epochs = 5#00
max_lbfgs_iterations = 1#0000

# Other parameters
best_model_working_period = 15.0
worst_model_working_period = 0.001

# Directory under 'results' and 'models' where the plots and models are going to be saved
results_and_models_subdirectory = 'one_tank'

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

# Load train and validation data
best_model_df = pd.read_csv('data/one_tank/rand_seed_30_t_range_10.0s_550_scenarios_200_collocation_points.csv')
worst_model_df = pd.read_csv('data/one_tank/rand_seed_30_t_range_0.001s_1100_scenarios_2_collocation_points.csv')

# Train data
best_model_train_df = best_model_df[best_model_df['scenario'] <= 500]
best_model_train_u_df = best_model_train_df[best_model_train_df['t'] == 0.0].sample(frac=1)
best_model_np_train_u_X = best_model_train_u_df[['t', 'v', 'ic']].to_numpy()
best_model_np_train_u_Y = best_model_train_u_df[['h']].to_numpy()
best_model_np_train_f_X = best_model_train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

worst_model_train_df = worst_model_df[worst_model_df['scenario'] <= 1000]
worst_model_train_u_df = worst_model_train_df[worst_model_train_df['t'] == 0.0].sample(frac=1)
worst_model_np_train_u_X = worst_model_train_u_df[['t', 'v', 'ic']].to_numpy()
worst_model_np_train_u_Y = worst_model_train_u_df[['h']].to_numpy()
worst_model_np_train_f_X = worst_model_train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

# Validation data
best_model_val_df = best_model_df[best_model_df['scenario'] > 500].sample(frac=1)
best_model_np_val_X = best_model_val_df[['t', 'v', 'ic']].to_numpy()
best_model_np_val_Y = best_model_val_df[['h']].to_numpy()

worst_model_val_df = worst_model_df[worst_model_df['scenario'] > 1000].sample(frac=1)
worst_model_np_val_X = worst_model_val_df[['t', 'v', 'ic']].to_numpy()
worst_model_np_val_Y = worst_model_val_df[['h']].to_numpy()

# Test data
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_t_range_160.0s_160000_collocation_points.csv')
np_test_t = test_df['t'].to_numpy()
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_h = test_df[['h']].to_numpy()
np_test_ic = np.array([test_df['h'].to_numpy()[0]])

# Tester
tester = BestAndWorstModelTester(OneTankPINN,
                                 best_model_hidden_layers, best_model_units_per_layer, best_model_working_period,
                                 worst_model_hidden_layers, worst_model_units_per_layer, worst_model_working_period,
                                 adam_epochs, max_lbfgs_iterations, sys_params)

# Load data into a container
data_container = BestAndWorstModelTestContainer()
best_model_key = tester.best_model_key
worst_model_key = tester.worst_model_key

data_container.set_train_u_X(best_model_key, best_model_np_train_u_X)
data_container.set_train_u_Y(best_model_key, best_model_np_train_u_Y)
data_container.set_train_f_X(best_model_key, best_model_np_train_f_X)

data_container.set_val_X(best_model_key, best_model_np_val_X)
data_container.set_val_Y(best_model_key, best_model_np_val_Y)

data_container.set_train_u_X(worst_model_key, worst_model_np_train_u_X)
data_container.set_train_u_Y(worst_model_key, worst_model_np_train_u_Y)
data_container.set_train_f_X(worst_model_key, worst_model_np_train_f_X)

data_container.set_val_X(worst_model_key, worst_model_np_val_X)
data_container.set_val_Y(worst_model_key, worst_model_np_val_Y)

data_container.test_t = np_test_t
data_container.test_X = np_test_X
data_container.test_Y = np_test_h
data_container.test_ic = np_test_ic

# Test
tester.test(data_container, results_and_models_subdirectory)
