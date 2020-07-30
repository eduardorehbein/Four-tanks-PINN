import numpy as np
import datetime
from util.normalizer import Normalizer
from util.plot import PdfPlotter


class StructTester:
    def __init__(self, layers_to_test, neurons_per_layer_to_test, adam_epochs=500, max_lbfgs_iterations=1000):
        self.layers_to_test = layers_to_test
        self.neurons_per_layer_to_test = neurons_per_layer_to_test
        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, PINNModelClass, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
             results_subdirectory, sys_params=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('Neural network\'s structural test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain Nu -> ' + str(np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(np_val_X.shape[0]))

        # Structural test
        plot_dict = dict()
        for layers in self.layers_to_test:
            plot_dict[layers] = {'final train u losses': [],
                                 'final train f losses': [],
                                 'final train total losses': [],
                                 'final val losses': []}

            for neurons in self.neurons_per_layer_to_test:
                # Instance PINN
                if sys_params is None:
                    model = PINNModelClass(layers, neurons, X_normalizer, Y_normalizer)
                else:
                    model = PINNModelClass(sys_params, layers, neurons, X_normalizer, Y_normalizer)

                # Train
                print('Model training with ' + str(layers) + ' hidden layers of ' + str(neurons) + ' neurons:')
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                            adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

                # Save plot data
                plot_dict[layers]['final train u losses'].append(model.train_u_loss[-1])
                plot_dict[layers]['final train f losses'].append(model.train_f_loss[-1])
                plot_dict[layers]['final train total losses'].append(model.train_total_loss[-1])
                plot_dict[layers]['final val losses'].append(model.validation_loss[-1])

        # Plot results
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final val losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final validation loss',
                     x_label='Neurons per layer',
                     y_label='Loss [u²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train total losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train total loss',
                     x_label='Neurons per layer',
                     y_label='Loss [u²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train u losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train u loss',
                     x_label='Neurons per layer',
                     y_label='Loss [u²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train f losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train f loss',
                     x_label='Neurons per layer',
                     y_label='Loss [u²]',
                     y_scale='log')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')


class WorkingPeriodTester:
    def __init__(self, working_periods, adam_epochs=500, max_lbfgs_iterations=1000):
        self.working_periods = working_periods

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, PINNModelClass, hidden_layers, units_per_layer, data_container,
             results_subdirectory, sys_params=None):
        nu = data_container.get_train_u_X(self.working_periods[0]).shape[0]
        nf = data_container.get_train_f_X(self.working_periods[0]).shape[0]
        val_scenarios = data_container.get_val_X(self.working_periods[0]).shape[0]
        test_points = data_container.test_X.shape[0]

        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('Neural network\'s working period test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain Nu -> ' + str(nu) +
                          '\nTrain Nf -> ' + str(nf) +
                          '\nValidation points -> ' + str(val_scenarios) +
                          '\nTest points -> ' + str(test_points))

        plot_dict = {'final train u losses': [], 'final train f losses': [],
                     'final train total losses': [], 'final val losses': [],
                     't': data_container.test_t, 'y': data_container.test_Y,
                     'nns': [], 'titles': []}

        for working_period in self.working_periods:
            # Train data
            np_train_u_X = data_container.get_train_u_X(working_period)
            np_train_u_Y = data_container.get_train_u_Y(working_period)
            np_train_f_X = data_container.get_train_f_X(working_period)

            # Validation data
            np_val_X = data_container.get_val_X(working_period)
            np_val_Y = data_container.get_val_Y(working_period)

            # Normalizers
            X_normalizer = Normalizer()
            Y_normalizer = Normalizer()

            X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
            Y_normalizer.parametrize(np_train_u_Y)

            # Instance PINN
            if sys_params is None:
                model = PINNModelClass(hidden_layers, units_per_layer, X_normalizer, Y_normalizer)
            else:
                model = PINNModelClass(sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer)

            # Train
            print('Model training with working period of ' + str(working_period) + ' seconds:')
            model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                        adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

            # Test
            nn = model.predict(data_container.test_X, data_container.test_ic, working_period)
            plot_dict['nns'].append(nn)

            plot_dict['titles'].append('Working period = ' + str(round(working_period, 3)) + ' s.')

            # Final losses
            plot_dict['final train u losses'].append(model.train_u_loss[-1])
            plot_dict['final train f losses'].append(model.train_f_loss[-1])
            plot_dict['final train total losses'].append(model.train_total_loss[-1])
            plot_dict['final val losses'].append(model.validation_loss[-1])

        # Plot losses
        np_working_periods = np.array(self.working_periods)
        plotter.plot(x_axis=np_working_periods,
                     y_axis_list=[np.array(plot_dict['final train total losses']),
                                  np.array(plot_dict['final val losses'])],
                     labels=['train loss', 'val loss'],
                     title='Train and validation total losses',
                     x_label='Working period [s]',
                     y_label='Loss [u²]',
                     y_scale='log')
        plotter.plot(x_axis=np_working_periods,
                     y_axis_list=[np.array(plot_dict['final train u losses']),
                                  np.array(plot_dict['final train f losses'])],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='Working period [s]',
                     y_label='Loss [u²]',
                     y_scale='log')

        # Plot test results
        for nn, title in zip(plot_dict['nns'], plot_dict['titles']):
            transposed_nn = np.transpose(nn)
            transposed_y = np.transpose(plot_dict['y'])
            index = 0
            for current_nn, current_y in zip(transposed_nn, transposed_y):
                index += 1
                mse = (np.square(current_y - current_nn)).mean()
                plotter.plot(x_axis=plot_dict['t'],
                             y_axis_list=[current_y, current_nn],
                             labels=['y' + str(index), 'nn' + str(index)],
                             title=title + ' Plot MSE: ' + str(round(mse, 3)) + ' u',
                             x_label='Time [s]',
                             y_label='Output [u]')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-working-period-test.pdf')


class NfNuTester:
    def __init__(self, nfs_to_test, nus_to_test, adam_epochs=500, max_lbfgs_iterations=1000):
        self.nfs_to_test = nfs_to_test
        self.nus_to_test = nus_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, PINNModelClass, hidden_layers, units_per_layer, data_container,
             results_subdirectory, sys_params=None):
        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('Neural network\'s Nf/Nu test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(hidden_layers) +
                          ' hidden layers of ' + str(units_per_layer) + ' neurons' +
                          '\nValidation points -> 10% of Nf')

        # Test
        plot_dict = dict()
        for nf in self.nfs_to_test:
            plot_dict[nf] = {'final train u losses': [],
                             'final train f losses': [],
                             'final train total losses': [],
                             'final val losses': []}

            for nu in self.nus_to_test:
                # Train data
                np_train_u_X = data_container.get_train_u_X(nf, nu)
                np_train_u_Y = data_container.get_train_u_Y(nf, nu)
                np_train_f_X = data_container.get_train_f_X(nf, nu)

                # Validation data
                np_val_X = data_container.get_val_X(nf, nu)
                np_val_Y = data_container.get_val_Y(nf, nu)

                # Normalizers
                X_normalizer = Normalizer()
                Y_normalizer = Normalizer()

                X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
                Y_normalizer.parametrize(np_train_u_Y)

                # Instance PINN
                if sys_params is None:
                    model = PINNModelClass(hidden_layers, units_per_layer, X_normalizer, Y_normalizer)
                else:
                    model = PINNModelClass(sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer)

                # Train
                print('Model training with Nu = ' + str(nu) + ' and Nf = ' + str(nf) + ':')
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                            adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

                # Save plot data
                plot_dict[nf]['final train u losses'].append(model.train_u_loss[-1])
                plot_dict[nf]['final train f losses'].append(model.train_f_loss[-1])
                plot_dict[nf]['final train total losses'].append(model.train_total_loss[-1])
                plot_dict[nf]['final val losses'].append(model.validation_loss[-1])

        # Plot results
        plotter.plot(x_axis=np.array(self.nus_to_test),
                     y_axis_list=[np.array(plot_dict[nf]['final val losses']) for nf in self.nfs_to_test],
                     labels=['Nf = ' + str(nf) for nf in self.nfs_to_test],
                     title='Final validation loss',
                     x_label='Nu',
                     y_label='Loss [u²]',
                     x_scale='log',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.nus_to_test),
                     y_axis_list=[np.array(plot_dict[nf]['final train total losses']) for nf in self.nfs_to_test],
                     labels=['Nf = ' + str(nf) for nf in self.nfs_to_test],
                     title='Final train total loss',
                     x_label='Nu',
                     y_label='Loss [u²]',
                     x_scale='log',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.nus_to_test),
                     y_axis_list=[np.array(plot_dict[nf]['final train u losses']) for nf in self.nfs_to_test],
                     labels=['Nf = ' + str(nf) for nf in self.nfs_to_test],
                     title='Final train u loss',
                     x_label='Nu',
                     y_label='Loss [u²]',
                     x_scale='log',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.nus_to_test),
                     y_axis_list=[np.array(plot_dict[nf]['final train f losses']) for nf in self.nfs_to_test],
                     labels=['Nf = ' + str(nf) for nf in self.nfs_to_test],
                     title='Final train f loss',
                     x_label='Nu',
                     y_label='Loss [u²]',
                     x_scale='log',
                     y_scale='log')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test.pdf')


class BestAndWorstModelTester:
    def __init__(self, adam_epochs=500, max_lbfgs_iterations=10000):
        self.best_model_key = 'best'
        self.worst_model_key = 'worst'

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, PINNModelClass, data_container, best_model_hidden_layers, best_model_units_per_layer,
             worst_model_hidden_layers, worst_model_units_per_layer, results_and_models_subdirectory, sys_params=None):
        # Train data
        best_model_np_train_u_X = data_container.get_train_u_X(self.best_model_key)
        best_model_np_train_u_Y = data_container.get_train_u_Y(self.best_model_key)
        best_model_np_train_f_X = data_container.get_train_f_X(self.best_model_key)

        worst_model_np_train_u_X = data_container.get_train_u_X(self.worst_model_key)
        worst_model_np_train_u_Y = data_container.get_train_u_Y(self.worst_model_key)
        worst_model_np_train_f_X = data_container.get_train_f_X(self.worst_model_key)

        # Validation data
        best_model_np_val_X = data_container.get_val_X(self.best_model_key)
        best_model_np_val_Y = data_container.get_val_Y(self.best_model_key)

        worst_model_np_val_X = data_container.get_val_X(self.worst_model_key)
        worst_model_np_val_Y = data_container.get_val_Y(self.worst_model_key)

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
        if sys_params is None:
            best_model = PINNModelClass(best_model_hidden_layers, best_model_units_per_layer,
                                        best_model_X_normalizer, best_model_Y_normalizer)
            worst_model = PINNModelClass(worst_model_hidden_layers, worst_model_units_per_layer,
                                         worst_model_X_normalizer, worst_model_Y_normalizer)
        else:
            best_model = PINNModelClass(sys_params, best_model_hidden_layers, best_model_units_per_layer,
                                        best_model_X_normalizer, best_model_Y_normalizer)
            worst_model = PINNModelClass(sys_params, worst_model_hidden_layers, worst_model_units_per_layer,
                                         worst_model_X_normalizer, worst_model_Y_normalizer)

        # Train
        best_model.train(best_model_np_train_u_X, best_model_np_train_u_Y, best_model_np_train_f_X,
                         best_model_np_val_X, best_model_np_val_Y, self.adam_epochs, self.max_lbfgs_iterations)

        worst_model.train(worst_model_np_train_u_X, worst_model_np_train_u_Y, worst_model_np_train_f_X,
                          worst_model_np_val_X, worst_model_np_val_Y, self.adam_epochs, self.max_lbfgs_iterations)

        # Load test data
        np_test_X = data_container.test_X
        np_test_Y = data_container.test_Y
        test_ic = data_container.test_ic

        # Test
        np_best_model_prediction = best_model.predict(np_test_X, np_ic=test_ic, working_period=1.0)
        np_worst_model_prediction = worst_model.predict(np_test_X, np_ic=test_ic, working_period=8.0)

        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('Van der Pol best and worst model:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTest points -> ' + str(np_test_X.shape[0]))

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
        np_t = data_container.test_t
        np_test_U = data_container.get_test_U()
        plotter.plot(x_axis=np_t,
                     y_axis_list=[np_u for np_u in np.transpose(np_test_U)],
                     labels=['u' + str(i + 1) for i in range(np_test_U.shape[1])],
                     title='Input signal',
                     x_label='Time [s]',
                     y_label=['Input [u]'])
        for i in range(np_test_Y.shape[1]):
            plotter.plot(x_axis=np_t,
                         y_axis_list=[np_best_model_prediction[:, i], np_worst_model_prediction[:, i], np_test_Y[:, i]],
                         labels=['Best model', 'Worst model', 'Casadi simulator'],
                         title='Output ' + str(i + 1) + ' prediction',
                         x_label='Time [s]',
                         y_label='Output [u]')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_and_models_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-worst-model-test.pdf')

        # Save models
        best_model.save_weights('models/' + results_and_models_subdirectory + '/' +
                                now.strftime('%Y-%m-%d-%H-%M-%S') + '-best-model.h5')
        worst_model.save_weights('models/' + results_and_models_subdirectory + '/' +
                                 now.strftime('%Y-%m-%d-%H-%M-%S') + '-worst-model.h5')


class WorkingPeriodTestContainer:
    def __init__(self):
        self.train_val_data = dict()
        self.test_t = None
        self.test_X = None
        self.test_Y = None
        self.test_ic = None

    def check_key(self, working_period):
        if working_period not in self.train_val_data.keys():
            self.train_val_data[working_period] = dict()

    def get_train_u_X(self, woking_period):
        return self.train_val_data[woking_period]['np_train_u_X']

    def get_train_u_Y(self, woking_period):
        return self.train_val_data[woking_period]['np_train_u_Y']

    def get_train_f_X(self, woking_period):
        return self.train_val_data[woking_period]['np_train_f_X']

    def get_val_X(self, woking_period):
        return self.train_val_data[woking_period]['np_val_X']

    def get_val_Y(self, woking_period):
        return self.train_val_data[woking_period]['np_val_Y']

    def set_train_u_X(self, woking_period, np_train_u_X):
        self.check_key(woking_period)
        self.train_val_data[woking_period]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, woking_period, np_train_u_Y):
        self.check_key(woking_period)
        self.train_val_data[woking_period]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, woking_period, np_train_f_X):
        self.check_key(woking_period)
        self.train_val_data[woking_period]['np_train_f_X'] = np_train_f_X

    def set_val_X(self, woking_period, np_val_X):
        self.check_key(woking_period)
        self.train_val_data[woking_period]['np_val_X'] = np_val_X

    def set_val_Y(self, woking_period, np_val_Y):
        self.check_key(woking_period)
        self.train_val_data[woking_period]['np_val_Y'] = np_val_Y


class NfNuTestContainer:
    def __init__(self):
        self.data = dict()

    def check_key(self, nf, nu):
        if nf not in self.data.keys():
            self.data[nf] = dict()
            self.data[nf][nu] = dict()
        elif nu not in self.data[nf].keys():
            self.data[nf][nu] = dict()

    def get_train_u_X(self, nf, nu):
        return self.data[nf][nu]['np_train_u_X']

    def get_train_u_Y(self, nf, nu):
        return self.data[nf][nu]['np_train_u_Y']

    def get_train_f_X(self, nf, nu):
        return self.data[nf][nu]['np_train_f_X']

    def get_val_X(self, nf, nu):
        return self.data[nf][nu]['np_val_X']

    def get_val_Y(self, nf, nu):
        return self.data[nf][nu]['np_val_Y']

    def set_train_u_X(self, nf, nu, np_train_u_X):
        self.check_key(nf, nu)
        self.data[nf][nu]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, nf, nu, np_train_u_Y):
        self.check_key(nf, nu)
        self.data[nf][nu]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, nf, nu, np_train_f_X):
        self.check_key(nf, nu)
        self.data[nf][nu]['np_train_f_X'] = np_train_f_X

    def set_val_X(self, nf, nu, np_val_X):
        self.check_key(nf, nu)
        self.data[nf][nu]['np_val_X'] = np_val_X

    def set_val_Y(self, nf, nu, np_val_Y):
        self.check_key(nf, nu)
        self.data[nf][nu]['np_val_Y'] = np_val_Y


class BestAndWorstModelTestContainer:
    def __init__(self):
        self.keys = ['best', 'worst']
        self.train_val_data = dict([(key, {'np_train_u_Y': None,
                                           'np_train_u_X': None,
                                           'np_train_f_X': None,
                                           'np_val_X': None,
                                           'np_val_Y': None}) for key in self.keys])

        self.test_t = None
        self.test_X = None
        self.test_Y = None
        self.test_ic = None

    def check_key(self, key):
        if key not in self.keys:
            raise Exception('Model parameter has to be in ' + str(self.keys))

    def get_train_u_X(self, model):
        return self.train_val_data[model]['np_train_u_X']

    def get_train_u_Y(self, model):
        return self.train_val_data[model]['np_train_u_Y']

    def get_train_f_X(self, model):
        return self.train_val_data[model]['np_train_f_X']

    def get_val_X(self, model):
        return self.train_val_data[model]['np_val_X']

    def get_val_Y(self, model):
        return self.train_val_data[model]['np_val_Y']

    def set_train_u_X(self, model, np_train_u_X):
        self.check_key(model)
        self.train_val_data[model]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, model, np_train_u_Y):
        self.check_key(model)
        self.train_val_data[model]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, model, np_train_f_X):
        self.check_key(model)
        self.train_val_data[model]['np_train_f_X'] = np_train_f_X

    def set_val_X(self, model, np_val_X):
        self.check_key(model)
        self.train_val_data[model]['np_val_X'] = np_val_X

    def set_val_Y(self, model, np_val_Y):
        self.check_key(model)
        self.train_val_data[model]['np_val_Y'] = np_val_Y

    def get_test_U(self):
        if self.test_X is None:
            raise Exception('Container\'s test_X data not defined.')
        elif np.array_equal(self.test_X.flatten(), self.test_t.flatten()):
            raise Exception('Container\'s test_X equals test_t, so there is no u defined.')
        else:
            i = 0
            while not np.array_equal(self.test_X[:, i].flatten(), self.test_t.flatten()):
                i = i + 1
            return np.delete(self.test_X, i, axis=1)
