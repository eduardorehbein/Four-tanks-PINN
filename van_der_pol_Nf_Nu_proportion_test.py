import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.tests import NfNuTester
from util.data_container import NfNuTestContainer
from util.data_interface import TrainDataGenerator

# Nf/Nu proportion test's parameters
nfs_to_test = (2000, 4000, 10000, 100000)
nus_to_test = (40, 80, 100, 500, 1000)

# Neural network's parameters
hidden_layers = 4
units_per_layer = 20

# Train data parameters
np_lowest_u = np.array([-1.0])
np_highest_u = np.array([1.0])
np_lowest_x = np.array([-3.0, -3.0])
np_highest_x = np.array([3.0, 3.0])

# Train parameters
adam_epochs = 500
max_lbfgs_iterations = 2000
train_T = 1.0
val_T = 0.5

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
data_container = NfNuTestContainer()
data_container.train_T = train_T

# Validation data
val_df = pd.read_csv('data/van_der_pol/rand_seed_60_sim_time_10.0s_10_scenarios_200_collocation_points.csv')
data_container.np_val_X = val_df[['t', 'u']].to_numpy()
data_container.np_val_Y = val_df[['x1', 'x2']].to_numpy()
data_container.np_val_ic = val_df[val_df['t'] == 0.0][['x1', 'x2']].to_numpy()
data_container.val_T = val_T

# Train data generator
train_data_gen = TrainDataGenerator(np_lowest_u=np_lowest_u,
                                    np_highest_u=np_highest_u,
                                    np_lowest_y=np_lowest_x,
                                    np_highest_y=np_highest_x)

for nf in nfs_to_test:
    for nu in nus_to_test:
        # Train data
        np_train_u_X, np_train_u_Y, np_train_f_X = train_data_gen.get_data(nu, int(nf/nu), train_T)
        data_container.set_train_u_X(nf, nu, np_train_u_X)
        data_container.set_train_u_Y(nf, nu, np_train_u_Y)
        data_container.set_train_f_X(nf, nu, np_train_f_X)

# Test
tester = NfNuTester(VanDerPolPINN, hidden_layers, units_per_layer, nfs_to_test, nus_to_test,
                    adam_epochs, max_lbfgs_iterations, random_seed=random_seed)
tester.test(data_container, results_subdirectory)
