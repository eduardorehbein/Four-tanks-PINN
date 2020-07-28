import numpy as np
import tensorflow as tf
import pandas as pd
from util.tests import WorkingPeriodTester, WorkingPeriodTestContainer
from util.pinn import VanDerPolPINN

# Working period test's parameters
working_periods_to_test = (0.01, 0.1, 1.0, 2.0, 4.0, 8.0)

# Neural network's parameters
hidden_layers = 5
units_per_layer = 20

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 2000

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'van_der_pol'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# Load data into a container
data_container = WorkingPeriodTestContainer()

test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_30_t_range_10.0s_10000_collocation_points.csv')
data_container.test_t = test_df['t'].to_numpy()
data_container.test_X = test_df[['t', 'u']].to_numpy()
data_container.test_Y = test_df[['x1', 'x2']].to_numpy()
data_container.test_ic = np.array(test_df[['x1', 'x2']].to_numpy()[0])

for working_period in working_periods_to_test:
    df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_' + str(working_period) +
                     's_1100_scenarios_100_collocation_points.csv')

    # Train data
    train_df = df[df['scenario'] <= 1000]
    train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
    np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
    np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
    np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

    data_container.set_train_u_X(working_period, np_train_u_X)
    data_container.set_train_u_Y(working_period, np_train_u_Y)
    data_container.set_train_f_X(working_period, np_train_f_X)

    # Validation data
    val_df = df[df['scenario'] > 1000].sample(frac=1)
    np_val_X = val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
    np_val_Y = val_df[['x1', 'x2']].to_numpy()

    data_container.set_val_X(working_period, np_val_X)
    data_container.set_val_Y(working_period, np_val_Y)

# Test
tester = WorkingPeriodTester(working_periods_to_test, adam_epochs, max_lbfgs_iterations)
tester.test(VanDerPolPINN, hidden_layers, units_per_layer, data_container, results_subdirectory)
