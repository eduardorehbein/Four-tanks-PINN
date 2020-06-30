import numpy as np
import tensorflow as tf
import datetime
from util.systems.four_tanks_system import CasadiSimulator
from util.normalizer import Normalizer
from util.pinn import FourTanksPINN
from util.plot import PdfPlotter

# Parallel threads config
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

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
train_points = 1000
np_train_vs = np.random.uniform(low=0.5, high=3.0, size=(2, train_points))
np_train_ics = np.random.uniform(low=2.0, high=20.0, size=(4, train_points))

validation_points = 100
np_validation_vs = np.random.uniform(low=0.5, high=3.0, size=(2, validation_points))
np_validation_ics = np.random.uniform(low=2.0, high=20.0, size=(4, validation_points))

test_points = 5
np_test_vs = np.random.uniform(low=0.5, high=3.0, size=(2, test_points))
np_test_ics = np.random.uniform(low=2.0, high=20.0, size=(4, test_points))

# Neural network's working period
t_range = 15.0
np_t = np.array([np.linspace(0, t_range, 100)])

# Train data
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
for i in range(np_validation_vs.shape[1]):
    np_v = np.transpose(np.tile(np_validation_vs[:, i], (np_t.shape[1], 1)))
    np_ic = np.transpose(np.tile(np_validation_ics[:, i], (np_t.shape[1], 1)))
    np_h = simulator.run(np_t[0], np_validation_vs[:, i], np_validation_ics[:, i])

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
                      X_normalizer=X_normalizer,
                      Y_normalizer=Y_normalizer)

# Training
model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y)

# Testing
sampled_outputs = []
predictions = []
titles = []

for i in range(np_test_vs.shape[1]):
    np_v = np_test_vs[:, i]
    np_ic = np_test_ics[:, i]

    np_h = simulator.run(np_t[0], np_v, np_ic)
    sampled_outputs.append(np_h)

    np_test_v = np.transpose(np.tile(np_v, (np_t.shape[1], 1)))
    np_test_ic = np.transpose(np.tile(np_ic, (np_t.shape[1], 1)))

    np_test_X = np.transpose(np.concatenate([np_t, np_test_v, np_test_ic], axis=0))

    prediction = model.predict(np_test_X)
    predictions.append(np.transpose(prediction))

    title = 'Control input v = (' + str(round(np_v[0], 2)) + ', ' + str(round(np_v[1], 2)) + \
            ') V.'
    titles.append(title)

# Plotter
plotter = PdfPlotter()

# Loss plot
plotter.plot(x_axis=np.linspace(1, len(model.train_total_loss), len(model.train_total_loss)),
             y_axis_list=[np.array(model.train_total_loss), np.array(model.validation_loss)],
             labels=['train loss', 'val loss'],
             title='Train and validation total losses',
             x_label='Epoch',
             y_label='Loss [cm²]',
             limit_range=False,
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, len(model.train_u_loss), len(model.train_u_loss)),
             y_axis_list=[np.array(model.train_u_loss), np.array(model.train_f_loss)],
             labels=['u loss', 'f loss'],
             title='Train losses',
             x_label='Epoch',
             y_label='Loss [cm²]',
             limit_range=False,
             y_scale='log')

# Result plot
for i in range(test_points):
    for j in range(sampled_outputs[i].shape[0]):
        y_axis_list = [sampled_outputs[i][j], predictions[i][j]]
        plotter.set_y_range(y_axis_list)
    for j in range(sampled_outputs[i].shape[0]):
        y_axis_list = [sampled_outputs[i][j], predictions[i][j]]
        mse = (np.square(y_axis_list[0] - y_axis_list[1])).mean()
        plotter.plot(x_axis=np_t[0],
                     y_axis_list=y_axis_list,
                     labels=['h' + str(j + 1), 'nn' + str(j + 1)],
                     title=titles[i] + ' Plot MSE: ' + str(round(mse, 3)) + ' cm²',
                     x_label='Time [s]',
                     y_label='Level [cm]',
                     limit_range=True)
now = datetime.datetime.now()
plotter.save_pdf('./results/four_tanks/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.pdf')

# Model saving
model.save_weights('./models/four_tanks/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.h5')
