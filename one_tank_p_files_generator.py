import pandas as pd
import numpy as np
from util.pinn import OneTankPINN
from util.file_generator import PFileGenerator


# Directory where the p files will be saved
directory = 'p_files/one_tank'

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14,  # [cm^3/Vs]
              }

# Data loading
df = pd.read_csv('data/one_tank/rand_seed_30_t_range_10.0s_550_scenarios_200_collocation_points.csv')

# Train data
train_df = df[df['scenario'] <= 500]
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'v', 'ic']].to_numpy()
np_train_u_Y = train_u_df[['h']].to_numpy()
np_train_f_X = train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

# Validation data
val_df = df[df['scenario'] > 500].sample(frac=1)
np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
np_val_Y = val_df[['h']].to_numpy()

# Test data
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_sim_time_160.0s_1600_collocation_points.csv')
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_Y = test_df[['h']].to_numpy()

# Instance PINN
model = OneTankPINN(hidden_layers=2,
                    units_per_layer=10,
                    sys_params=sys_params)

# Load model
model.load('models/one_tank/2020-08-06-11-24-21-10s-2l-10n-best-model')

# Weights p file
generator = PFileGenerator()
generator.gen_train_data_file(np_train_u_X, np_train_u_Y, np_train_f_X, directory)
generator.gen_val_data_file(np_val_X, np_val_Y, directory)
generator.gen_test_data_file(np_test_X, np_test_Y, directory)
