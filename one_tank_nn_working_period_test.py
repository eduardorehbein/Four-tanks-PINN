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
working_periods_to_test = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
dfs = [pd.read_csv('data/one_tank/rand_seed_30_t_range_' + str(working_period) +
                   's_1105_scenarios_100_collocation_points.csv') for working_period in working_periods_to_test]

# Train parameters
max_adam_epochs = 500
max_lbfgs_iterations = 1000

# Plotter
plotter = PdfPlotter()
plotter.text_page('One tank neural network\'s working period test:' +
                  '\nAdam epochs -> ' + str(max_adam_epochs) +
                  '\nL-BFGS iterations -> ' + str(max_lbfgs_iterations) +
                  '\nTrain Nu -> 1e3' +
                  '\nTrain Nf -> 100e3' +
                  '\nValidation points -> 10e3' +
                  '\nTest points -> ' + str(100*len(working_periods_to_test)))

plot_dict = {'final train u losses': [], 'final train f losses': [],
             'final train total losses': [], 'final val losses': [],
             'ts': [], 'hs': [], 'nns': [], 'titles': []}

for df in dfs:
    # Current working period
    working_period = df['t'].max()

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
    print('Model training with working period of ' + str(working_period) + ' seconds:')
    model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                max_adam_epochs=max_adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

    # Test
    test_df = df[df['scenario'] == 1101]

    plot_dict['ts'].append(test_df['t'].to_numpy())
    plot_dict['hs'].append(test_df['h'].to_numpy())

    np_test_X = test_df[['t', 'v', 'ic']].to_numpy()
    nn = model.predict(np_test_X)
    plot_dict['nns'].append(nn)

    v = test_df['v'].min()
    title = 'Control input v = ' + str(round(v, 3)) + ' V.'
    plot_dict['titles'].append(title)

    # Final losses
    plot_dict['final train u losses'].append(model.train_u_loss[-1])
    plot_dict['final train f losses'].append(model.train_f_loss[-1])
    plot_dict['final train total losses'].append(model.train_total_loss[-1])
    plot_dict['final val losses'].append(model.validation_loss[-1])

# Plot losses
plotter.plot(x_axis=np.array(working_periods_to_test),
             y_axis_list=[np.array(plot_dict['final train total losses']), np.array(plot_dict['final val losses'])],
             labels=['train loss', 'val loss'],
             title='Train and validation total losses',
             x_label='Working period [s]',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.array(working_periods_to_test),
             y_axis_list=[np.array(plot_dict['final train u losses']), np.array(plot_dict['final train f losses'])],
             labels=['u loss', 'f loss'],
             title='Train losses',
             x_label='Working period [s]',
             y_label='Loss [cm²]',
             y_scale='log')

# Plot test results
y_axis_list = np.concatenate(plot_dict['hs'] + plot_dict['nns'])
plotter.set_y_range(y_axis_list)
for t, h, nn, title in zip(plot_dict['ts'], plot_dict['hs'], plot_dict['nns'], plot_dict['titles']):
    mse = (np.square(h - nn)).mean()
    plotter.plot(x_axis=t,
                 y_axis_list=[h, nn],
                 labels=['h', 'nn'],
                 title=title + ' Plot MSE: ' + str(round(mse, 3)) + ' cm²',
                 x_label='Time [s]',
                 y_label='Level [cm]',
                 limit_range=True)

# Save results
now = datetime.datetime.now()
plotter.save_pdf('./results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-working-period-test.pdf')
