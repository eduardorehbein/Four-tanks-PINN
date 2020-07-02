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

# Load train and validation data
best_model_df = pd.read_csv('data/one_tank/rand_seed_30_t_range_10.0s_550_scenarios_200_collocation_points.csv')
worst_model_df = pd.read_csv('data/one_tank/rand_seed_30_t_range_0.001s_1100_scenarios_2_collocation_points.csv')

# Train data
best_model_train_df = best_model_df[best_model_df['scenario'] <= 500]
best_model_train_u_df = best_model_train_df[best_model_train_df['t'] == 0.0].sample(frac=1)
best_model_np_train_u_X = best_model_train_u_df[['t', 'v', 'ic']].to_numpy()
best_model_np_train_u_Y = best_model_train_u_df[['h']].to_numpy()
best_model_np_train_f_X = best_model_train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

worst_model_train_df = worst_model_df[worst_model_df['scenario'] <= 1000]
worst_model_train_u_df = worst_model_train_df[worst_model_train_df['t'] == 0.0].sample(frac=1)
worst_model_np_train_u_X = worst_model_train_u_df[['t', 'v', 'ic']].to_numpy()
worst_model_np_train_u_Y = worst_model_train_u_df[['h']].to_numpy()
worst_model_np_train_f_X = worst_model_train_df[['t', 'v', 'ic']].sample(frac=1).to_numpy()

# Validation data
best_model_val_df = best_model_df[(best_model_df['scenario'] > 500) &
                                  (best_model_df['scenario'] <= 550)].sample(frac=1)
best_model_np_val_X = best_model_val_df[['t', 'v', 'ic']].to_numpy()
best_model_np_val_Y = best_model_val_df[['h']].to_numpy()

worst_model_val_df = worst_model_df[(worst_model_df['scenario'] > 1000) &
                                    (worst_model_df['scenario'] <= 1100)].sample(frac=1)
worst_model_np_val_X = worst_model_val_df[['t', 'v', 'ic']].to_numpy()
worst_model_np_val_Y = worst_model_val_df[['h']].to_numpy()

# Normalizers
best_model_X_normalizer = Normalizer()
best_model_Y_normalizer = Normalizer()

best_model_X_normalizer.parametrize(np.concatenate([best_model_np_train_u_X, best_model_np_train_f_X], axis=0))
best_model_Y_normalizer.parametrize(best_model_np_train_u_Y)

worst_model_X_normalizer = Normalizer()
worst_model_Y_normalizer = Normalizer()

worst_model_X_normalizer.parametrize(np.concatenate([worst_model_np_train_u_X, worst_model_np_train_f_X], axis=0))
worst_model_Y_normalizer.parametrize(worst_model_np_train_u_Y)

# Instance PINN
best_model = OneTankPINN(sys_params=sys_params,
                         hidden_layers=2,
                         units_per_layer=10,
                         X_normalizer=best_model_X_normalizer,
                         Y_normalizer=best_model_Y_normalizer)

worst_model = OneTankPINN(sys_params=sys_params,
                          hidden_layers=2,
                          units_per_layer=10,
                          X_normalizer=worst_model_X_normalizer,
                          Y_normalizer=worst_model_Y_normalizer)

# Train
max_adam_epochs = 500
max_lbfgs_iterations = 10000

best_model.train(best_model_np_train_u_X, best_model_np_train_u_Y, best_model_np_train_f_X,
                 best_model_np_val_X, best_model_np_val_Y,
                 max_adam_epochs=max_adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

worst_model.train(worst_model_np_train_u_X, worst_model_np_train_u_Y, worst_model_np_train_f_X,
                  worst_model_np_val_X, worst_model_np_val_Y,
                  max_adam_epochs=max_adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

# Load test data
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_t_range_160.0s_160000_collocation_points.csv')
np_test_X = test_df[['t', 'v']].to_numpy()
np_test_h = test_df[['h']].to_numpy()
test_ic = np.array([test_df['h'].to_numpy()[0]])

# Test
np_best_model_prediction = best_model.predict(np_test_X, np_ic=test_ic, working_period=10.0)
np_worst_model_prediction = worst_model.predict(np_test_X, np_ic=test_ic, working_period=0.001)

# Plotter
plotter = PdfPlotter()
plotter.text_page('One tank best and worst model:' +
                  '\nAdam epochs -> ' + str(max_adam_epochs) +
                  '\nL-BFGS iterations -> ' + str(max_lbfgs_iterations) +
                  '\nNeural networks\' structure -> 2 hidden layers of 10 neurons' +
                  '\nTest points -> ' + str(len(test_df)))

# Plot train and validation losses
loss_len = min(len(best_model.train_total_loss), len(worst_model.train_total_loss))
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.validation_loss[:loss_len]),
                          np.array(worst_model.validation_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Validation loss',
             x_label='Epoch',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_total_loss[:loss_len]),
                          np.array(worst_model.train_total_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train total loss',
             x_label='Epoch',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_u_loss[:loss_len]),
                          np.array(worst_model.train_u_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train u loss',
             x_label='Epoch',
             y_label='Loss [cm²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_f_loss[:loss_len]),
                          np.array(worst_model.train_f_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train f loss',
             x_label='Epoch',
             y_label='Loss [cm²]',
             y_scale='log')

# Plot test results
np_t = test_df['t'].to_numpy()
plotter.plot(x_axis=np_t,
             y_axis_list=[test_df['v'].to_numpy()],
             labels=['Input'],
             title='Input signal',
             x_label='Time [s]',
             y_label='Valve opening [V]')
plotter.plot(x_axis=np_t,
             y_axis_list=[np_best_model_prediction, np_worst_model_prediction, np_test_h],
             labels=['Best model', 'Worst model', 'Casadi simulator'],
             title='Output prediction',
             x_label='Time [s]',
             y_label='Level [cm]')

# Save results
now = datetime.datetime.now()
plotter.save_pdf('results/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-worst-model-test.pdf')

# Save models
best_model.save_weights('models/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-model.h5')
worst_model.save_weights('models/one_tank/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-worst-model.h5')
