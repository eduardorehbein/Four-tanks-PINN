import numpy as np
import datetime
from one_tank_system import CasadiSimulator
from normalizer import Normalizer
from pinn import OneTankPINN
from plot import PdfPlotter

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

# Training data
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
    np_h = simulator.run(np.transpose(np_t), np_validation_vs[i, 0], np_validation_ics[i, 0])

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

# PINN instancing
model = OneTankPINN(sys_params=sys_params,
                    hidden_layers=2,
                    units_per_layer=15,
                    X_normalizer=X_normalizer,
                    Y_normalizer=Y_normalizer)

# Training
model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y, max_epochs=40000)

# Testing
sampled_outputs = []
predictions = []
titles = []
round_in_title = 3

for i in range(np_test_vs.shape[0]):
    np_v = np_test_vs[i, 0]
    np_ic = np_test_ics[i, 0]

    np_h = simulator.run(np.transpose(np_t), np_v, np_ic)
    sampled_outputs.append(np_h)

    np_test_v = np.tile(np_v, (np_t.shape[0], 1))
    np_test_ic = np.tile(np_ic, (np_t.shape[0], 1))

    np_test_X = np.concatenate([np_t, np_test_v, np_test_ic], axis=1)

    prediction = model.predict(np_test_X)
    predictions.append(np.transpose(prediction))

    title = 'Control input v = ' + str(round(np_v, round_in_title)) + ' V.'
    titles.append(title)

# Plotter
plotter = PdfPlotter()

# Loss plot
plotter.plot(x_axis=np.linspace(1, len(model.train_total_loss), len(model.train_total_loss)),
             y_axis_list=[np.array(model.train_total_loss), np.array(model.validation_loss)],
             labels=['train loss', 'val loss'],
             title='Train and validation total losses',
             x_label='Epoch',
             y_label='Loss',
             limit_range=False,
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, len(model.train_u_loss), len(model.train_u_loss)),
             y_axis_list=[np.array(model.train_u_loss), np.array(model.train_f_loss)],
             labels=['u loss', 'f loss'],
             title='Train losses',
             x_label='Epoch',
             y_label='Loss',
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
        plotter.plot(x_axis=np.transpose(np_t)[0],
                     y_axis_list=y_axis_list,
                     labels=['h', 'nn'],
                     title=titles[i] + ' Plot MSE: ' + str(round(mse, round_in_title)),
                     x_label='t',
                     y_label='Level',
                     limit_range=True)
now = datetime.datetime.now()
plotter.save_pdf('./results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.pdf')

# Model saving
model.save_weights('./models/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.h5')
