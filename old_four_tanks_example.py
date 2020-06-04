import numpy as np
import datetime
from four_tanks_system import ResponseAnalyser, CasadiSimulator
from normalizer import Normalizer
from pinn import OldFourTanksPINN
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
resp_an = ResponseAnalyser(sys_params)
# TODO: Improve parameter analysis method
# t_range = resp_an.get_ol_sample_time(np.concatenate([np_train_vs, np_test_vs], axis=1))
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

    if np_train_f_t is None:
        np_train_f_t = np_t
    else:
        np_train_f_t = np.append(np_train_f_t, np_t, axis=1)
    if np_train_f_v is None:
        np_train_f_v = np_v
    else:
        np_train_f_v = np.append(np_train_f_v, np_v, axis=1)
    if np_train_f_ic is None:
        np_train_f_ic = np_ic
    else:
        np_train_f_ic = np.append(np_train_f_ic, np_ic, axis=1)

# Validation data
np_validation_t = None
np_validation_v = None
np_validation_ic = None
np_validation_h = None

simulator = CasadiSimulator(sys_params)
for i in range(np_validation_vs.shape[1]):
    np_v = np.transpose(np.tile(np_validation_vs[:, i], (np_t.shape[1], 1)))
    np_ic = np.transpose(np.tile(np_validation_ics[:, i], (np_t.shape[1], 1)))
    np_h = simulator.run(np_t, np_validation_vs[:, i], np_validation_ics[:, i])

    if np_validation_t is None:
        np_validation_t = np_t
    else:
        np_validation_t = np.append(np_validation_t, np_t, axis=1)
    if np_validation_v is None:
        np_validation_v = np_v
    else:
        np_validation_v = np.append(np_validation_v, np_v, axis=1)
    if np_validation_ic is None:
        np_validation_ic = np_ic
    else:
        np_validation_ic = np.append(np_validation_ic, np_ic, axis=1)
    if np_validation_h is None:
        np_validation_h = np_h
    else:
        np_validation_h = np.append(np_validation_h, np_h, axis=1)

# Normalizers
t_normalizer = Normalizer()
v_normalizer = Normalizer()
h_normalizer = Normalizer()

t_normalizer.parametrize(np_t)
v_normalizer.parametrize(np_train_vs)
h_normalizer.parametrize(np_train_ics)

# Train data normalization
np_norm_train_u_t = t_normalizer.normalize(np_train_u_t)
np_norm_train_u_v = v_normalizer.normalize(np_train_u_v)
np_norm_train_u_ic = h_normalizer.normalize(np_train_u_ic)

np_norm_train_f_t = t_normalizer.normalize(np_train_f_t)
np_norm_train_f_v = v_normalizer.normalize(np_train_f_v)
np_norm_train_f_ic = h_normalizer.normalize(np_train_f_ic)

# Validation data normalization
np_norm_validation_t = t_normalizer.normalize(np_validation_t)
np_norm_validation_v = v_normalizer.normalize(np_validation_v)
np_norm_validation_ic = h_normalizer.normalize(np_validation_ic)
np_norm_validation_h = h_normalizer.normalize(np_validation_h)

# PINN instancing
hidden_layers = [15, 15, 15, 15, 15]
learning_rate = 0.001
model = OldFourTanksPINN(sys_params=sys_params,
                         hidden_layers=hidden_layers,
                         learning_rate=learning_rate,
                         t_normalizer=t_normalizer,
                         v_normalizer=v_normalizer,
                         h_normalizer=h_normalizer)

# Training
max_epochs = 40000
stop_loss = 0.0005
model.train(np_train_u_t=np_norm_train_u_t,
            np_train_u_v=np_norm_train_u_v,
            np_train_u_ic=np_norm_train_u_ic,
            np_train_f_t=np_norm_train_f_t,
            np_train_f_v=np_norm_train_f_v,
            np_train_f_ic=np_norm_train_f_ic,
            np_validation_t=np_norm_validation_t,
            np_validation_v=np_norm_validation_v,
            np_validation_ic=np_norm_validation_ic,
            np_validation_h=np_norm_validation_h,
            max_epochs=max_epochs,
            stop_loss=stop_loss)

# Testing
sampled_outputs = []
predictions = []
titles = []

np_norm_t = t_normalizer.normalize(np_t)
for i in range(np_test_vs.shape[1]):
    np_v = np_test_vs[:, i]
    np_ic = np_test_ics[:, i]

    np_h = simulator.run(np_t, np_v, np_ic)
    sampled_outputs.append(np_h)

    np_test_v = np.transpose(np.tile(np_v, (np_t.shape[1], 1)))
    np_test_ic = np.transpose(np.tile(np_ic, (np_t.shape[1], 1)))

    np_norm_test_v = v_normalizer.normalize(np_test_v)
    np_norm_test_ic = h_normalizer.normalize(np_test_ic)

    norm_prediction = model.predict(np_norm_t, np_norm_test_v, np_norm_test_ic)
    prediction = h_normalizer.denormalize(norm_prediction)
    predictions.append(prediction)

    title = 'Control input v = (' + str(round(np_v[0], 2)) + ', ' + str(round(np_v[1], 2)) + \
            ') V.'
    titles.append(title)

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
number_of_plots = test_points
for i in range(number_of_plots):
    for j in range(sampled_outputs[i].shape[0]):
        y_axis_list = [sampled_outputs[i][j], predictions[i][j]]
        plotter.set_y_range(y_axis_list)
    for j in range(sampled_outputs[i].shape[0]):
        y_axis_list = [sampled_outputs[i][j], predictions[i][j]]
        mse = (np.square(y_axis_list[0] - y_axis_list[1])).mean()
        plotter.plot(x_axis=np_t[0],
                     y_axis_list=y_axis_list,
                     labels=['h' + str(j + 1), 'nn' + str(j + 1)],
                     title=titles[i] + ' Plot MSE: ' + str(round(mse, 2)),
                     x_label='t',
                     y_label='Level',
                     limit_range=True)
now = datetime.datetime.now()
plotter.save_pdf('./results/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.pdf')
