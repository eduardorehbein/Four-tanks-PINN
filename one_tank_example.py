import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from util.normalizer import Normalizer
from util.pinn import OneTankPINN
from util.plot import PdfPlotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Load data
df = pd.read_csv('data/one_tank/rand_seed_30_t_range_10.0s_1105_scenarios_100_collocation_points.csv')

# Train data
train_df = df[df['scenario'] <= 1000]
train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
np_train_u_X = train_u_df[['t', 'v', 'ic']].to_numpy()
np_train_u_Y = train_u_df[['h']].to_numpy()
np_train_f_X = train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

# Validation data
val_df = df[(df['scenario'] > 1000) & (df['scenario'] <= 1100)].sample(frac=1)
np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
np_val_Y = val_df[['h']].to_numpy()

# Normalizers
X_normalizer = Normalizer()
Y_normalizer = Normalizer()

X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X], axis=0))
Y_normalizer.parametrize(np_train_u_Y)

# Instance PINN
model = OneTankPINN(sys_params=sys_params,
                    hidden_layers=2,
                    units_per_layer=10,
                    X_normalizer=X_normalizer,
                    Y_normalizer=Y_normalizer)

# Train
model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y, max_lbfgs_iterations=10000)

# Test
sampled_outputs = []
predictions = []
titles = []
round_in_title = 3

scenarios = df['scenario'].max()
for scenario in range(scenarios - 4, scenarios + 1):
    test_df = df[df['scenario'] == scenario]

    sampled_outputs.append(test_df['h'].to_numpy())

    np_test_X = test_df[['t', 'v', 'ic']].to_numpy()
    prediction = model.predict(np_test_X)
    predictions.append(prediction)

    v = test_df['v'].min()
    title = 'Control input v = ' + str(round(v, round_in_title)) + ' V.'
    titles.append(title)

# Plotter
plotter = PdfPlotter()

# Plot losses
train_total_loss_len = len(model.train_total_loss)
plotter.plot(x_axis=np.linspace(1, train_total_loss_len, train_total_loss_len),
             y_axis_list=[np.array(model.train_total_loss), np.array(model.validation_loss)],
             labels=['train loss', 'val loss'],
             title='Train and validation total losses',
             x_label='Epoch',
             y_label='Loss [cm²]',
             limit_range=False,
             y_scale='log')
train_u_loss_len = len(model.train_u_loss)
plotter.plot(x_axis=np.linspace(1, train_u_loss_len, train_u_loss_len),
             y_axis_list=[np.array(model.train_u_loss), np.array(model.train_f_loss)],
             labels=['u loss', 'f loss'],
             title='Train losses',
             x_label='Epoch',
             y_label='Loss [cm²]',
             limit_range=False,
             y_scale='log')

# Plot test results
y_axis_list = np.concatenate(sampled_outputs + predictions)
plotter.set_y_range(y_axis_list)
np_t = df[df['scenario'] == 1]['t'].to_numpy()
for h, nn, title in zip(sampled_outputs, predictions, titles):
    mse = (np.square(h - nn)).mean()
    plotter.plot(x_axis=np_t,
                 y_axis_list=[h, nn],
                 labels=['h', 'nn'],
                 title=title + ' Plot MSE: ' + str(round(mse, round_in_title)) + ' cm²',
                 x_label='Time [s]',
                 y_label='Level [cm]',
                 limit_range=True)

# Save results
now = datetime.datetime.now()
plotter.save_pdf('results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.pdf')

# Save model
model.save_weights('models/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.h5')
