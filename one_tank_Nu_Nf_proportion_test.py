import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from util.normalizer import Normalizer
from util.pinn import OneTankPINN
from util.plot import PdfPlotter

# Parallel threads config
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

# Test parameters
nfs_to_test = (2000, 4000, 10000, 100000)
nus_to_test = (40, 70, 100, 500, 1000)

max_adam_epochs = 500
max_lbfgs_iterations = 1000

# Plotter
plotter = PdfPlotter()
plotter.text_page('One tank neural network\'s Nu/Nf test:' +
                  '\nAdam epochs -> ' + str(max_adam_epochs) +
                  '\nL-BFGS iterations -> ' + str(max_lbfgs_iterations) +
                  '\nNeural network\'s T -> 10s' +
                  '\nNeural network\'s structure -> 2 hidden layers of 10 neurons' +
                  '\nValidation points -> 10% of Nf')

# Structural test
plot_dict = dict()
for nf in nfs_to_test:
    plot_dict[nf] = {'final train u losses': [],
                     'final train f losses': [],
                     'final train total losses': [],
                     'final val losses': []}

    for nu in nus_to_test:
        # Data loading
        df = pd.read_csv('data/one_tank/rand_seed_30_t_range_10.0s_' + str(int(1.1*nu)) +
                         '_scenarios_' + str(int(nf/nu)) + '_collocation_points.csv')

        # Train data
        train_df = df[df['scenario'] <= nu]
        train_u_df = train_df[train_df['t'] == 0.0].sample(frac=1)
        np_train_u_X = train_u_df[['t', 'v', 'ic']].to_numpy()
        np_train_u_Y = train_u_df[['h']].to_numpy()
        np_train_f_X = train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

        # Validation data
        val_df = df[(df['scenario'] > nu) & (df['scenario'] <= int(1.1*nu))].sample(frac=1)
        np_val_X = val_df[['t', 'v', 'ic']].to_numpy()
        np_val_Y = val_df[['h']].to_numpy()

        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X], axis=0))
        Y_normalizer.parametrize(np_train_u_Y)

        # PINN instance
        model = OneTankPINN(sys_params=sys_params,
                            hidden_layers=2,
                            units_per_layer=10,
                            X_normalizer=X_normalizer,
                            Y_normalizer=Y_normalizer)

        # Train
        print('Model training with Nu = ' + str(nu) + ' and Nf = ' + str(nf) + ':')
        model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                    max_adam_epochs=max_adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

        # Save plot data
        plot_dict[nf]['final train u losses'].append(model.train_u_loss[-1])
        plot_dict[nf]['final train f losses'].append(model.train_f_loss[-1])
        plot_dict[nf]['final train total losses'].append(model.train_total_loss[-1])
        plot_dict[nf]['final val losses'].append(model.validation_loss[-1])

plotter.plot(x_axis=np.array(nus_to_test),
             y_axis_list=[np.array(plot_dict[nf]['final val losses']) for nf in nfs_to_test],
             labels=['Nf = ' + str(nf) for nf in nfs_to_test],
             title='Final validation loss',
             x_label='Nu',
             y_label='Loss [cm²]',
             x_scale='log',
             y_scale='log')
plotter.plot(x_axis=np.array(nus_to_test),
             y_axis_list=[np.array(plot_dict[nf]['final train total losses']) for nf in nfs_to_test],
             labels=['Nf = ' + str(nf) for nf in nfs_to_test],
             title='Final train total loss',
             x_label='Nu',
             y_label='Loss [cm²]',
             x_scale='log',
             y_scale='log')
plotter.plot(x_axis=np.array(nus_to_test),
             y_axis_list=[np.array(plot_dict[nf]['final train u losses']) for nf in nfs_to_test],
             labels=['Nf = ' + str(nf) for nf in nfs_to_test],
             title='Final train u loss',
             x_label='Nu',
             y_label='Loss [cm²]',
             x_scale='log',
             y_scale='log')
plotter.plot(x_axis=np.array(nus_to_test),
             y_axis_list=[np.array(plot_dict[nf]['final train f losses']) for nf in nfs_to_test],
             labels=['Nf = ' + str(nf) for nf in nfs_to_test],
             title='Final train f loss',
             x_label='Nu',
             y_label='Loss [cm²]',
             x_scale='log',
             y_scale='log')

now = datetime.datetime.now()
plotter.save_pdf('results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nu-Nf-proportion-test.pdf')
