import pickle

import numpy as np
from util.systems.one_tank_system import CasadiSimulator
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

# Controls and initial conditions for training and testing
train_points = 1000
np_train_vs = np.random.uniform(low=0.5, high=3.0, size=(train_points, 1))
np_train_ics = np.random.uniform(low=2.0, high=20.0, size=(train_points, 1))

validation_points = 100
np_validation_vs = np.random.uniform(low=0.5, high=3.0, size=(validation_points, 1))
np_validation_ics = np.random.uniform(low=2.0, high=20.0, size=(validation_points, 1))

test_points = 5
np_test_vs = np.random.uniform(low=0.5, high=3.0, size=(test_points, 1))
np_test_ics = np.random.uniform(low=2.0, high=20.0, size=(test_points, 1))

# Neural network's working period
t_range = 15.0
np_t = np.transpose(np.array([np.linspace(0, t_range, 100)]))

# Train data
np_train_u_t = np.zeros((np_train_ics.shape[0], 1))
np_train_u_v = np_train_vs
np_train_u_ic = np_train_ics

np_train_f_t = None
np_train_f_v = None
np_train_f_ic = None

for i in range(np_train_vs.shape[0]):
    np_v = np.tile(np_train_vs[i, 0], (np_t.shape[0], 1))
    np_ic = np.tile(np_train_ics[i, 0], (np_t.shape[0], 1))

    if i == 0:
        np_train_f_t = np_t
        np_train_f_v = np_v
        np_train_f_ic = np_ic
    else:
        np_train_f_t = np.append(np_train_f_t, np_t, axis=0)
        np_train_f_v = np.append(np_train_f_v, np_v, axis=0)
        np_train_f_ic = np.append(np_train_f_ic, np_ic, axis=0)

np_train_u_X = np.concatenate([np_train_u_t, np_train_u_v, np_train_u_ic], axis=1)
np_train_u_Y = np_train_u_ic

np_train_f_X = np.concatenate([np_train_f_t, np_train_f_v, np_train_f_ic], axis=1)

# Train p file
train_data = [np_train_u_X, np_train_u_Y, np_train_f_X]
with open('p_files/one_tank/train_data.p', 'wb') as train_p_file:
    pickle.dump(train_data, train_p_file)

# Normalizers
X_normalizer = Normalizer()
Y_normalizer = Normalizer()

X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X], axis=0))
Y_normalizer.parametrize(np_train_u_Y)

# Validation data
np_val_t = None
np_val_v = None
np_val_ic = None
np_val_h = None

simulator = CasadiSimulator(sys_params)
for i in range(np_validation_vs.shape[0]):
    np_v = np.tile(np_validation_vs[i, 0], (np_t.shape[0], 1))
    np_ic = np.tile(np_validation_ics[i, 0], (np_t.shape[0], 1))
    np_h = simulator.run(np_t[0, :], np_validation_vs[i, 0], np_validation_ics[i, 0])

    if i == 0:
        np_val_t = np_t
        np_val_v = np_v
        np_val_ic = np_ic
        np_val_h = np.transpose(np_h)
    else:
        np_val_t = np.append(np_val_t, np_t, axis=0)
        np_val_v = np.append(np_val_v, np_v, axis=0)
        np_val_ic = np.append(np_val_ic, np_ic, axis=0)
        np_val_h = np.append(np_val_h, np.transpose(np_h), axis=0)

np_val_X = np.concatenate([np_val_t, np_val_v, np_val_ic], axis=1)
np_val_Y = np_val_h

# Validation p file
validation_data = [np_val_X, np_val_Y]
with open('p_files/one_tank/validation_data.p', 'wb') as validation_p_file:
    pickle.dump(validation_data, validation_p_file)

# PINN instancing
model = OneTankPINN(sys_params=sys_params,
                    hidden_layers=2,
                    units_per_layer=15,
                    X_normalizer=X_normalizer,
                    Y_normalizer=Y_normalizer)

# Model loading
model.load_weights('./models/one_tank/2020-06-09-10-41-32-2l-15n.h5')

# Weights p file
weights = model.get_weights()
with open('p_files/one_tank/weights.p', 'wb') as weights_p_file:
    pickle.dump(weights, weights_p_file)

# Test data
np_test_t = None
np_test_v = None
np_test_ic = None
np_test_h = None

for i in range(np_test_vs.shape[0]):
    np_v = np.tile(np_test_vs[i, 0], (np_t.shape[0], 1))
    np_ic = np.tile(np_test_ics[i, 0], (np_t.shape[0], 1))
    np_h = simulator.run(np_t[0, :], np_test_vs[i, 0], np_test_ics[i, 0])

    if i == 0:
        np_test_t = np_t
        np_test_v = np_v
        np_test_ic = np_ic
        np_test_h = np.transpose(np_h)
    else:
        np_test_t = np.append(np_test_t, np_t, axis=0)
        np_test_v = np.append(np_test_v, np_v, axis=0)
        np_test_ic = np.append(np_test_ic, np_ic, axis=0)
        np_test_h = np.append(np_test_h, np.transpose(np_h), axis=0)

np_test_X = np.concatenate([np_test_t, np_test_v, np_test_ic], axis=1)
np_test_Y = np_test_h

# Test p file
test_data = [np_test_X, np_test_Y]
with open('p_files/one_tank/test_data.p', 'wb') as test_p_file:
    pickle.dump(test_data, test_p_file)
