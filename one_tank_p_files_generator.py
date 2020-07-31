import pickle
import pandas as pd
import numpy as np
from util.normalizer import Normalizer
from util.pinn import OneTankPINN

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

# Train p file
train_data = [np_train_u_X, np_train_u_Y, np_train_f_X]
with open('p_files/one_tank/train_data.p', 'wb') as train_p_file:
    pickle.dump(train_data, train_p_file)

# Normalizers
X_normalizer = Normalizer()
Y_normalizer = Normalizer()

X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
Y_normalizer.parametrize(np_train_u_Y)

# Validation data
val_df = df[df['scenario'] > 500].sample(frac=1)
np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
np_val_Y = val_df[['h']].to_numpy()

# Validation p file
validation_data = [np_val_X, np_val_Y]
with open('p_files/one_tank/validation_data.p', 'wb') as validation_p_file:
    pickle.dump(validation_data, validation_p_file)

# Instance PINN
model = OneTankPINN(sys_params=sys_params,
                    hidden_layers=2,
                    units_per_layer=10,
                    X_normalizer=X_normalizer,
                    Y_normalizer=Y_normalizer)

# Load model
model.load_weights('models/one_tank/2020-07-02-10-25-19-10s-2l-10n-best-model.h5')

# Weights p file
weights = model.get_weights()
with open('p_files/one_tank/weights.p', 'wb') as weights_p_file:
    pickle.dump(weights, weights_p_file)

# Test data
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_t_range_160.0s_1600_collocation_points.csv')
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_Y = test_df[['h']].to_numpy()

# Test p file
test_data = [np_test_X, np_test_Y]
with open('p_files/one_tank/test_data.p', 'wb') as test_p_file:
    pickle.dump(test_data, test_p_file)
