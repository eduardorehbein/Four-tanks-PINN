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
df = pd.read_csv('data/one_tank/rand_seed_30_t_range_15.0s_2000_scenarios_100_collocation_points.csv')

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

# Test parameters
layers_to_test = (1, 2, 3)
neurons_per_layer_to_test = (2, 3, 5, 8, 10, 15, 20)

max_adam_epochs = 500
max_lbfgs_iterations = 1000

# Plotter
plotter = PdfPlotter()
plotter.text_page('One tank neural network\'s structural test:' +
                  '\nAdam epochs -> ' + str(max_adam_epochs) +
                  '\nL-BFGS iterations -> ' + str(max_lbfgs_iterations) +
                  '\nTrain Nu -> 1e3' +
                  '\nTrain Nf -> 100e3' +
                  '\nValidation points -> 10e3')

# Structural test
plot_dict = dict()
for layers in layers_to_test:
    plot_dict[layers] = {'final train u losses': [],
                         'final train f losses': [],
                         'final train total losses': [],
                         'final val losses': []}

    for neurons in neurons_per_layer_to_test:
        # Instance PINN
        model = OneTankPINN(sys_params=sys_params,
                            hidden_layers=layers,
                            units_per_layer=neurons,
                            X_normalizer=X_normalizer,
                            Y_normalizer=Y_normalizer)

        # Train
        print('Model training with ' + str(layers) + ' hidden layers of ' + str(neurons) + ' neurons:')
        model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                    max_adam_epochs=max_adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

        # Save plot data
        plot_dict[layers]['final train u losses'].append(model.train_u_loss[-1])
        plot_dict[layers]['final train f losses'].append(model.train_f_loss[-1])
        plot_dict[layers]['final train total losses'].append(model.train_total_loss[-1])
        plot_dict[layers]['final val losses'].append(model.validation_loss[-1])

# Plot results
plotter.plot(x_axis=np.array(neurons_per_layer_to_test),
             y_axis_list=[np.array(plot_dict[layers]['final val losses']) for layers in layers_to_test],
             labels=[str(layers) + ' layers' for layers in layers_to_test],
             title='Final validation loss',
             x_label='Neurons per layer',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.array(neurons_per_layer_to_test),
             y_axis_list=[np.array(plot_dict[layers]['final train total losses']) for layers in layers_to_test],
             labels=[str(layers) + ' layers' for layers in layers_to_test],
             title='Final train total loss',
             x_label='Neurons per layer',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.array(neurons_per_layer_to_test),
             y_axis_list=[np.array(plot_dict[layers]['final train u losses']) for layers in layers_to_test],
             labels=[str(layers) + ' layers' for layers in layers_to_test],
             title='Final train u loss',
             x_label='Neurons per layer',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.array(neurons_per_layer_to_test),
             y_axis_list=[np.array(plot_dict[layers]['final train f losses']) for layers in layers_to_test],
             labels=[str(layers) + ' layers' for layers in layers_to_test],
             title='Final train f loss',
             x_label='Neurons per layer',
             y_label='Loss [cm²]',
             y_scale='log')

# Save results
now = datetime.datetime.now()
plotter.save_pdf('results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')
