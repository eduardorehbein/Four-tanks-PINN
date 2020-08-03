import pickle
import pandas as pd
import numpy as np
from util.normalizer import Normalizer
from util.pinn import VanDerPolPINN


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

# Train p file
train_data = [np_train_u_X, np_train_u_Y, np_train_f_X]
with open('p_files/van_der_pol/train_data.p', 'wb') as train_p_file:
    pickle.dump(train_data, train_p_file)

# Normalizers
X_normalizer = Normalizer()
Y_normalizer = Normalizer()

X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
Y_normalizer.parametrize(np_train_u_Y)

# Validation data
val_df = df[df['scenario'] > 100].sample(frac=1)
np_val_X = val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
np_val_Y = val_df[['x1', 'x2']].to_numpy()

# Validation p file
validation_data = [np_val_X, np_val_Y]
with open('p_files/van_der_pol/validation_data.p', 'wb') as validation_p_file:
    pickle.dump(validation_data, validation_p_file)

# Instance PINN
model = VanDerPolPINN(hidden_layers=5,
                      units_per_layer=20,
                      X_normalizer=X_normalizer,
                      Y_normalizer=Y_normalizer)

# Load model
model.load_weights('models/van_der_pol/2020-07-30-10-45-58-1s-5l-20n-best-model.h5')

# Weights p file
weights = model.get_weights()
with open('p_files/van_der_pol/weights.p', 'wb') as weights_p_file:
    pickle.dump(weights, weights_p_file)

# Test data
test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_30_t_range_10.0s_10000_collocation_points.csv')
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_Y = test_df[['x1', 'x2']].to_numpy()

# Test p file
test_data = [np_test_X, np_test_Y]
with open('p_files/van_der_pol/test_data.p', 'wb') as test_p_file:
    pickle.dump(test_data, test_p_file)
