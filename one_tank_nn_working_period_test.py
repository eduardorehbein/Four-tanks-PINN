import numpy as np
import tensorflow as tf
import pandas as pd
from util.tests import WorkingPeriodTester, WorkingPeriodTestContainer
from util.pinn import OneTankPINN

# Parameters
working_periods_to_test = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
train_scenarios = 1000
val_scenarios = 100

# Neural network's parameters
hidden_layers = 2
units_per_layer = 10

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'one_tank'

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 1000

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
random_seed = 30
np.random.seed(random_seed)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Load data into a container
data_container = WorkingPeriodTestContainer()

test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_' + str(random_seed) +
                      '_t_range_160.0s_1600_collocation_points.csv')
data_container.test_t = test_df['t'].to_numpy()
data_container.test_X = test_df[['t', 'v']].to_numpy()
data_container.test_Y = test_df[['h']].to_numpy()
data_container.test_ic = np.array([test_df['h'].to_numpy()[0]])

for working_period in working_periods_to_test:
    df = pd.read_csv('data/one_tank/rand_seed_' + str(random_seed) + '_t_range_' + str(working_period) +
                     's_1105_scenarios_100_collocation_points.csv')

    # Train data
    train_df = df[df['scenario'] <= train_scenarios]
    train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
    np_train_u_X = train_u_df[['t', 'v', 'ic']].to_numpy()
    np_train_u_Y = train_u_df[['h']].to_numpy()
    np_train_f_X = train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

    data_container.set_train_u_X(working_period, np_train_u_X)
    data_container.set_train_u_Y(working_period, np_train_u_Y)
    data_container.set_train_f_X(working_period, np_train_f_X)

    # Validation data
    val_df = df[(df['scenario'] > train_scenarios) &
                (df['scenario'] <= (train_scenarios + val_scenarios))].sample(frac=1)
    np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
    np_val_Y = val_df[['h']].to_numpy()

    data_container.set_val_X(working_period, np_val_X)
    data_container.set_val_Y(working_period, np_val_Y)

# Test
tester = WorkingPeriodTester(working_periods_to_test, adam_epochs, max_lbfgs_iterations)
tester.test(OneTankPINN, sys_params, hidden_layers, units_per_layer, data_container, results_subdirectory)
