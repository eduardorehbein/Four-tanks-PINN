import pandas as pd
import numpy as np
from util.pinn import VanDerPolPINN
from util.file_generator import PFileGenerator


# Directory where the p files will be saved
directory = 'p_files/van_der_pol'

# Random seed
np.random.seed(30)

# Data loading
df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_1.0s_110_scenarios_20_collocation_points.csv')

# Train data
train_df = df[df['scenario'] <= 100]
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_train_u_Y = train_u_df[['x1', 'x2']].to_numpy()
np_train_f_X = train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

# Validation data
val_df = df[df['scenario'] > 100].sample(frac=1)
np_val_X = val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_val_Y = val_df[['x1', 'x2']].to_numpy()

# Test data
test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_30_sim_time_10.0s_10000_collocation_points.csv')
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_Y = test_df[['x1', 'x2']].to_numpy()

# Instance PINN
model = VanDerPolPINN(hidden_layers=5,
                      units_per_layer=20)

# Load model
model.load('models/van_der_pol/2020-08-04-11-27-53-1s-5l-20n-best-model')

# P file
generator = PFileGenerator()
generator.gen_train_data_file(np_train_u_X, np_train_u_Y, np_train_f_X, directory)
generator.gen_val_data_file(np_val_X, np_val_Y, directory)
generator.gen_test_data_file(np_test_X, np_test_Y, directory)
