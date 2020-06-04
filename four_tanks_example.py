import numpy as np
import datetime
from four_tanks_system import ResponseAnalyser, CasadiSimulator
from pinn import FourTanksPINN
from plot import PdfPlotter

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a1': 0.071,  # [cm^2]
              'a2': 0.057,  # [cm^2]
              'a3': 0.071,  # [cm^2]
              'a4': 0.057,  # [cm^2]
              'A1': 28.0,  # [cm^2]
              'A2': 32.0,  # [cm^2]
              'A3': 28.0,  # [cm^2]
              'A4': 32.0,  # [cm^2]
              'alpha1': 0.5,  # [adm]
              'alpha2': 0.5,  # [adm]
              'k1': 3.33,  # [cm^3/Vs]
              'k2': 3.35,  # [cm^3/Vs]
              }

# Controls and initial conditions for training and testing
train_points = 950
np_train_vs = 3.0 * np.random.rand(2, train_points)
np_train_ics = 20.0 * np.random.rand(4, train_points)

validation_points = 50
np_validation_vs = 3.0 * np.random.rand(2, validation_points)
np_validation_ics = 20 * np.random.rand(4, validation_points)

test_points = 5
np_test_vs = 3.0 * np.random.rand(2, test_points)
np_test_ics = 20.0 * np.random.rand(4, test_points)

# Neural network's working period
t_range = 15.0
np_t = np.array([np.linspace(0, t_range, 100)])

# Training data
np_train_u_t = np.zeros((1, np_train_ics.shape[1]))
np_train_u_v = np_train_vs
np_train_u_ic = np_train_ics

np_train_f_t = None
np_train_f_v = None
np_train_f_ic = None

for i in range(np_train_vs.shape[1]):
    np_v = np.transpose(np.tile(np_train_vs[:, i], (np_t.shape[1], 1)))
    np_ic = np.transpose(np.tile(np_train_ics[:, i], (np_t.shape[1], 1)))

    if i == 0:
        np_train_f_t = np_t
        np_train_f_v = np_v
        np_train_f_ic = np_ic
    else:
        np_train_f_t = np.append(np_train_f_t, np_t, axis=1)
        np_train_f_v = np.append(np_train_f_v, np_v, axis=1)
        np_train_f_ic = np.append(np_train_f_ic, np_ic, axis=1)

np_train_u_X = np.transpose(np.concatenate([np_train_u_t, np_train_u_v, np_train_u_ic], axis=0))
np_train_u_Y = np.transpose(np_train_u_ic)

np_train_f_X = np.transpose(np.concatenate([np_train_f_t, np_train_f_v, np_train_f_ic], axis=0))

# Validation data
np_val_t = None
np_val_v = None
np_val_ic = None
np_val_h = None

simulator = CasadiSimulator(sys_params)
for i in range(np_validation_vs.shape[1]):
    np_v = np.transpose(np.tile(np_validation_vs[:, i], (np_t.shape[1], 1)))
    np_ic = np.transpose(np.tile(np_validation_ics[:, i], (np_t.shape[1], 1)))
    np_h = simulator.run(np_t, np_validation_vs[:, i], np_validation_ics[:, i])

    if i == 0:
        np_val_t = np_t
        np_val_v = np_v
        np_val_ic = np_ic
        np_val_h = np_h
    else:
        np_val_t = np.append(np_val_t, np_t, axis=1)
        np_val_v = np.append(np_val_v, np_v, axis=1)
        np_val_ic = np.append(np_val_ic, np_ic, axis=1)
        np_val_h = np.append(np_val_h, np_h, axis=1)

np_val_X = np.transpose(np.concatenate([np_val_t, np_val_v, np_val_ic], axis=0))
np_val_Y = np.transpose(np_val_h)

# PINN instancing
model = FourTanksPINN(sys_params=sys_params,
                      hidden_layers=5,
                      units_per_layer=15,
                      np_input_lower_bounds=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                      np_input_upper_bounds=np.array([[t_range, 3.0, 3.0, 20.0, 20.0, 20.0, 20.0]]),
                      np_output_lower_bounds=np.array([[0.0, 0.0, 0.0, 0.0]]),
                      np_output_upper_bounds=np.array([[20.0, 20.0, 20.0, 20.0]]))

# Training
model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y, max_epochs=40000)
