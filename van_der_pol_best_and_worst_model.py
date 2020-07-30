import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from util.normalizer import Normalizer
from util.pinn import VanDerPolPINN
from util.plot import PdfPlotter

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# Load train and validation data
best_model_df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_1.0s_110_scenarios_20_collocation_points.csv')
worst_model_df = pd.read_csv('data/van_der_pol/rand_seed_30_t_range_8.0s_1100_scenarios_4_collocation_points.csv')

# Train data
best_model_train_df = best_model_df[best_model_df['scenario'] <= 100]
best_model_train_u_df = best_model_train_df[best_model_train_df['t'] == 0.0].sample(frac=1)
best_model_np_train_u_X = best_model_train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
best_model_np_train_u_Y = best_model_train_u_df[['x1', 'x2']].to_numpy()
best_model_np_train_f_X = best_model_train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

worst_model_train_df = worst_model_df[worst_model_df['scenario'] <= 1000]
worst_model_train_u_df = worst_model_train_df[worst_model_train_df['t'] == 0.0].sample(frac=1)
worst_model_np_train_u_X = worst_model_train_u_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
worst_model_np_train_u_Y = worst_model_train_u_df[['x1', 'x2']].to_numpy()
worst_model_np_train_f_X = worst_model_train_df[['t', 'u', 'x1_0', 'x2_0']].sample(frac=1).to_numpy()

# Validation data
best_model_val_df = best_model_df[best_model_df['scenario'] > 100].sample(frac=1)
best_model_np_val_X = best_model_val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
best_model_np_val_Y = best_model_val_df[['x1', 'x2']].to_numpy()

worst_model_val_df = worst_model_df[worst_model_df['scenario'] > 1000].sample(frac=1)
worst_model_np_val_X = worst_model_val_df[['t', 'u', 'x1_0', 'x2_0']].to_numpy()
worst_model_np_val_Y = worst_model_val_df[['x1', 'x2']].to_numpy()

# Normalizers
best_model_X_normalizer = Normalizer()
best_model_Y_normalizer = Normalizer()

best_model_X_normalizer.parametrize(np.concatenate([best_model_np_train_u_X, best_model_np_train_f_X]))
best_model_Y_normalizer.parametrize(best_model_np_train_u_Y)

worst_model_X_normalizer = Normalizer()
worst_model_Y_normalizer = Normalizer()

worst_model_X_normalizer.parametrize(np.concatenate([worst_model_np_train_u_X, worst_model_np_train_f_X]))
worst_model_Y_normalizer.parametrize(worst_model_np_train_u_Y)

# Instance PINN
best_model = VanDerPolPINN(hidden_layers=5,
                           units_per_layer=20,
                           X_normalizer=best_model_X_normalizer,
                           Y_normalizer=best_model_Y_normalizer)

worst_model = VanDerPolPINN(hidden_layers=8,
                            units_per_layer=2,
                            X_normalizer=worst_model_X_normalizer,
                            Y_normalizer=worst_model_Y_normalizer)

# Train
adam_epochs = 500
max_lbfgs_iterations = 10000

best_model.train(best_model_np_train_u_X, best_model_np_train_u_Y, best_model_np_train_f_X,
                 best_model_np_val_X, best_model_np_val_Y,
                 adam_epochs=adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

worst_model.train(worst_model_np_train_u_X, worst_model_np_train_u_Y, worst_model_np_train_f_X,
                  worst_model_np_val_X, worst_model_np_val_Y,
                  adam_epochs=adam_epochs, max_lbfgs_iterations=max_lbfgs_iterations)

# Load test data
test_df = pd.read_csv('data/van_der_pol/long_signal_rand_seed_30_t_range_10.0s_10000_collocation_points.csv')
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_Y = test_df[['x1', 'x2']].to_numpy()
test_ic = np.array(test_df[['x1', 'x2']].to_numpy()[0])

# Test
np_best_model_prediction = best_model.predict(np_test_X, np_ic=test_ic, working_period=1.0)
np_worst_model_prediction = worst_model.predict(np_test_X, np_ic=test_ic, working_period=8.0)

# Plotter
plotter = PdfPlotter()
plotter.text_page('Van der Pol best and worst model:' +
                  '\nAdam epochs -> ' + str(adam_epochs) +
                  '\nL-BFGS iterations -> ' + str(max_lbfgs_iterations) +
                  '\nTest points -> ' + str(len(test_df)))

# Plot train and validation losses
loss_len = min(len(best_model.train_total_loss), len(worst_model.train_total_loss))
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.validation_loss[:loss_len]),
                          np.array(worst_model.validation_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Validation loss',
             x_label='Epoch',
             y_label='Loss [u²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_total_loss[:loss_len]),
                          np.array(worst_model.train_total_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train total loss',
             x_label='Epoch',
             y_label='Loss [u²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_u_loss[:loss_len]),
                          np.array(worst_model.train_u_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train u loss',
             x_label='Epoch',
             y_label='Loss [u²]',
             y_scale='log')
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=[np.array(best_model.train_f_loss[:loss_len]),
                          np.array(worst_model.train_f_loss[:loss_len])],
             labels=['Best model', 'Worst model'],
             title='Train f loss',
             x_label='Epoch',
             y_label='Loss [u²]',
             y_scale='log')

# Plot test results
np_t = test_df['t'].to_numpy()
plotter.plot(x_axis=np_t,
             y_axis_list=[test_df['u'].to_numpy()],
             labels=['Input'],
             title='Input signal',
             x_label='Time [s]',
             y_label='Input [u]')
for i in range(np_test_Y.shape[1]):
    plotter.plot(x_axis=np_t,
                 y_axis_list=[np_best_model_prediction[:, i], np_worst_model_prediction[:, i], np_test_Y[:, i]],
                 labels=['Best model', 'Worst model', 'Casadi simulator'],
                 title='Output ' + str(i + 1) + ' prediction',
                 x_label='Time [s]',
                 y_label='Output [u]')

# Save results
now = datetime.datetime.now()
plotter.save_pdf('results/van_der_pol/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-worst-model-test.pdf')

# Save models
best_model.save_weights('models/van_der_pol/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-model.h5')
worst_model.save_weights('models/van_der_pol/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-worst-model.h5')
