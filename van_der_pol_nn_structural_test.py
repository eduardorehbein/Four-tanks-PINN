import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.tests import StructTester

# Structural test parameters
layers_to_test = (4, 5, 8, 10)
neurons_per_layer_to_test = (2, 3, 5, 8, 10, 15, 20)

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

# Load data
df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_' + str(working_period) +
                 's_1100_scenarios_100_collocation_points.csv')

# Train data
train_df = df[df['scenario'] <= 1000]
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

# Validation data
val_df = df[df['scenario'] > 1000].sample(frac=1)
np_val_X = val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_val_Y = val_df[['x1', 'x2']].to_numpy()

# Test
tester = StructTester(layers_to_test, neurons_per_layer_to_test, adam_epochs, max_lbfgs_iterations)
tester.test(VanDerPolPINN, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
            results_subdirectory, working_period)
