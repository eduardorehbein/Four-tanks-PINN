import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.tests import NfNuTester, NfNuTestContainer

# Nf/Nu proportion test's parameters
nfs_to_test = (2000, 4000, 10000, 100000)
nus_to_test = (40, 70, 100, 500, 1000)

# Neural network's parameters
hidden_layers = 5
units_per_layer = 20

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 1000

# Other parameters
working_period = 1.0

# Directory under 'results' where the plots are going to be saved
results_subdirectory = 'van_der_pol'

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# Load data into a container
data_container = NfNuTestContainer()
for nf in nfs_to_test:
    for nu in nus_to_test:
        df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_' + str(working_period) + 's_' + str(int(1.1*nu)) +
                         '_scenarios_' + str(int(nf/nu)) + '_collocation_points.csv')

        # Train data
        train_df = df[df['scenario'] <= nu]
        train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
        np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
        np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
        np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

        data_container.set_train_u_X(nf, nu, np_train_u_X)
        data_container.set_train_u_Y(nf, nu, np_train_u_Y)
        data_container.set_train_f_X(nf, nu, np_train_f_X)

        # Validation data
        val_df = df[df['scenario'] > nu].sample(frac=1)
        np_val_X = val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
        np_val_Y = val_df[['x1', 'x2']].to_numpy()

        data_container.set_val_X(nf, nu, np_val_X)
        data_container.set_val_Y(nf, nu, np_val_Y)

# Test
tester = NfNuTester(VanDerPolPINN, hidden_layers, units_per_layer, nfs_to_test, nus_to_test, working_period,
                    adam_epochs, max_lbfgs_iterations)
tester.test(data_container, results_subdirectory)
