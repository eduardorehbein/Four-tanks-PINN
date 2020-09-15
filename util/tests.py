import numpy as np
import datetime
from util.normalizer import Normalizer
from util.plot import Plotter


class StructTester:
    def __init__(self, PINNModelClass, layers_to_test, neurons_per_layer_to_test,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.layers_to_test = layers_to_test
        self.neurons_per_layer_to_test = neurons_per_layer_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, T, np_val_Y,
             results_subdirectory=None, save_mode=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Plotter
        plotter = Plotter()
        plotter.text_page('Neural network\'s structural test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nT -> ' + str(T) + ' s' +
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
                if self.sys_params is None:
                    model = self.PINNModelClass(layers, neurons, X_normalizer, Y_normalizer)
                else:
                    model = self.PINNModelClass(self.sys_params, layers, neurons, X_normalizer, Y_normalizer)

                # Train
                print('Model training with ' + str(layers) + ' hidden layers of ' + str(neurons) + ' neurons:')
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, T, np_val_Y,
                            self.adam_epochs, self.max_lbfgs_iterations)

                # Save plot data
                plot_dict[layers]['final train u losses'].append(model.train_u_loss[-1])
                plot_dict[layers]['final train f losses'].append(model.train_f_loss[-1])
                plot_dict[layers]['final train total losses'].append(model.train_total_loss[-1])
                plot_dict[layers]['final val losses'].append(model.validation_loss[-1])

        # Plot results
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final val losses'])
                                           for layers in self.layers_to_test])),
                             title='Final validation losses (neurons x layers)',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train total losses'])
                                           for layers in self.layers_to_test])),
                             title='Final train total losses (neurons x layers)',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train u losses'])
                                           for layers in self.layers_to_test])),
                             title='Final train u losses (neurons x layers)',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train f losses'])
                                           for layers in self.layers_to_test])),
                             title='Final train f losses (neurons x layers)',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test)

        # Save or show results
        if save_mode == 'all':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test')
        elif save_mode == 'pdf':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')
        elif save_mode == 'eps':
            now = datetime.datetime.now()
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test')
        else:
            plotter.show()


class TTester:
    def __init__(self, PINNModelClass, hidden_layers, units_per_layer, Ts,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.Ts = Ts

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, data_container, results_subdirectory, save_mode=None):
        nu = data_container.get_train_u_X(self.Ts[0]).shape[0]
        nf = data_container.get_train_f_X(self.Ts[0]).shape[0]
        val_points = data_container.np_val_X.shape[0]
        test_points = data_container.np_test_X.shape[0]

        # Plotter
        plotter = Plotter()
        plotter.text_page('Neural network\'s T test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain Nu -> ' + str(nu) +
                          '\nTrain Nf -> ' + str(nf) +
                          '\nValidation points -> ' + str(val_points) +
                          '\nTest points -> ' + str(test_points))

        plot_dict = {'final train u losses': [], 'final train f losses': [],
                     'final train total losses': [], 'final val losses': [],
                     't': data_container.np_test_t, 'y': data_container.np_test_Y,
                     'nns': [], 'titles': []}

        for T in self.Ts:
            # Train data
            np_train_u_X = data_container.get_train_u_X(T)
            np_train_u_Y = data_container.get_train_u_Y(T)
            np_train_f_X = data_container.get_train_f_X(T)

            # Validation data
            np_val_X = data_container.np_val_X
            np_val_Y = data_container.np_val_Y
            np_val_ic = np_val_Y[0]

            # Normalizers
            X_normalizer = Normalizer()
            Y_normalizer = Normalizer()

            X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
            Y_normalizer.parametrize(np_train_u_Y)

            # Instance PINN
            if self.sys_params is None:
                model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer)
            else:
                model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                            X_normalizer, Y_normalizer)

            # Train
            print('Model training with T of ' + str(T) + ' seconds:')
            model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, T, np_val_Y,
                        adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

            # Test
            nn = model.predict(data_container.np_test_X, data_container.np_test_Y[0], T)
            plot_dict['nns'].append(nn)

            plot_dict['titles'].append('T = ' + str(round(T, 3)) + ' s.')

            # Final losses
            plot_dict['final train u losses'].append(model.train_u_loss[-1])
            plot_dict['final train f losses'].append(model.train_f_loss[-1])
            plot_dict['final train total losses'].append(model.train_total_loss[-1])
            plot_dict['final val losses'].append(model.validation_loss[-1])

        # Plot losses
        np_Ts = np.array(self.Ts)
        plotter.plot(x_axis=np_Ts,
                     y_axis_list=[np.array(plot_dict['final train total losses']),
                                  np.array(plot_dict['final val losses'])],
                     labels=['train loss', 'val loss'],
                     title='Train and validation total losses',
                     x_label='T',
                     y_label='Loss',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-')
        plotter.plot(x_axis=np_Ts,
                     y_axis_list=[np.array(plot_dict['final train u losses']),
                                  np.array(plot_dict['final train f losses'])],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='T',
                     y_label='Loss',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-')

        # Plot test results
        for nn, title, current_T in zip(plot_dict['nns'], plot_dict['titles'], np_Ts):
            transposed_nn = np.transpose(nn)
            transposed_y = np.transpose(plot_dict['y'])
            markevery = int(plot_dict['t'].size / (plot_dict['t'][-1] / current_T))
            output_index = 0
            for current_nn, current_y in zip(transposed_nn, transposed_y):
                output_index += 1
                mse = (np.square(current_y - current_nn)).mean()
                plotter.plot(x_axis=plot_dict['t'],
                             y_axis_list=[current_y, current_nn],
                             labels=['$\\hat{y}_{' + str(output_index) + '}$', '$y_{' + str(output_index) + '}$'],
                             title=title + ' MSE: ' + str(round(mse, 3)),
                             x_label='Time',
                             y_label='Output',
                             line_styles=['--', 'o-'],
                             markevery=markevery)

        # Save or show results
        if save_mode == 'all':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-T-test.pdf')
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-T-test')
        elif save_mode == 'pdf':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-T-test.pdf')
        elif save_mode == 'eps':
            now = datetime.datetime.now()
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-T-test')
        else:
            plotter.show()


class NfNuTester:
    def __init__(self, PINNModelClass, hidden_layers, units_per_layer, nfs_to_test, nus_to_test, T,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.T = T

        self.nfs_to_test = nfs_to_test
        self.nus_to_test = nus_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, data_container, results_subdirectory, save_mode=None):
        # Plotter
        plotter = Plotter()
        plotter.text_page('Neural network\'s Nf/Nu test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nWorking period -> ' + str(self.T) + ' s' +
                          '\nValidation points -> 10% of Nf')

        # Validation data
        np_val_X = data_container.np_val_X
        np_val_Y = data_container.np_val_Y
        np_val_ic = np_val_Y[0]

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

                # Normalizers
                X_normalizer = Normalizer()
                Y_normalizer = Normalizer()

                X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
                Y_normalizer.parametrize(np_train_u_Y)

                # Instance PINN
                if self.sys_params is None:
                    model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer)
                else:
                    model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                                X_normalizer, Y_normalizer)

                # Train
                print('Model training with Nu = ' + str(nu) + ' and Nf = ' + str(nf) + ':')
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, self.T, np_val_Y,
                            adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

                # Save plot data
                plot_dict[nf]['final train u losses'].append(model.train_u_loss[-1])
                plot_dict[nf]['final train f losses'].append(model.train_f_loss[-1])
                plot_dict[nf]['final train total losses'].append(model.train_total_loss[-1])
                plot_dict[nf]['final val losses'].append(model.validation_loss[-1])

        # Plot results
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[nf]['final val losses'])
                                                     for nf in self.nfs_to_test])),
                             title='Final validation losses $(N_t \\times N_f)$',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[nf]['final train total losses'])
                                                     for nf in self.nfs_to_test])),
                             title='Final train total losses $(N_t \\times N_f)$',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[nf]['final train u losses'])
                                                     for nf in self.nfs_to_test])),
                             title='Final train u losses $(N_t \\times N_f)$',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[nf]['final train f losses'])
                                                     for nf in self.nfs_to_test])),
                             title='Final train f losses $(N_t \\times N_f)$',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)

        # Save or show results
        if save_mode == 'all':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test.pdf')
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test')
        elif save_mode == 'pdf':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test.pdf')
        elif save_mode == 'eps':
            now = datetime.datetime.now()
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test')
        else:
            plotter.show()


class ExhaustionTester:
    def __init__(self, PINNModelClass, hidden_layers, units_per_layer,
                 adam_epochs=500, max_lbfgs_iterations=10000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, T, np_val_Y,
            np_test_t, np_test_X, np_test_ic, np_test_Y, results_and_models_subdirectory, save_mode=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Instance PINN
        if self.sys_params is None:
            model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer)
        else:
            model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                        X_normalizer, Y_normalizer)

        # Train
        model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_ic, T, np_val_Y,
                    self.adam_epochs, self.max_lbfgs_iterations)

        # Test
        model_prediction = model.predict(np_test_X, np_test_ic, T)

        # Plotter
        plotter = Plotter()
        plotter.text_page('Exhaustion test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nT -> ' + str(T) + ' s' +
                          '\nTrain Nu -> ' + str(np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(np_val_X.shape[0]) +
                          '\nTest points -> ' + str(np_test_X.shape[0]),
                          vertical_position=0.3)

        # Plot train and validation losses
        loss_len = len(model.train_total_loss)
        loss_x_axis = np.linspace(1, loss_len, loss_len)
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(model.train_total_loss), np.array(model.validation_loss)],
                     labels=['Train loss', 'Validation loss'],
                     title='Total losses',
                     x_label='Epoch',
                     y_label='Loss',
                     y_scale='log')
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(model.train_u_loss), np.array(model.train_f_loss)],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='Epoch',
                     y_label='Loss',
                     y_scale='log')

        # Plot test results
        t_index = 0
        while not np.array_equal(np_test_X[:, t_index].flatten(), np_test_t.flatten()):
            t_index = t_index + 1
        np_test_U = np.delete(np_test_X, t_index, axis=1)
        plotter.plot(x_axis=np_test_t,
                     y_axis_list=[np_test_U[:, i] for i in range(np_test_U.shape[1])],
                     labels=['$u_{' + str(i + 1) + '}$' for i in range(np_test_U.shape[1])],
                     title='Input signal',
                     x_label='Time',
                     y_label='Input')
        for i in range(np_test_Y.shape[1]):
            markevery = int(np_test_t.size / (np_test_t[-1] / T))
            mse = (np.square(model_prediction[:, i] - np_test_Y[:, i])).mean()
            plotter.plot(x_axis=np_test_t,
                         y_axis_list=[np_test_Y[:, i], model_prediction[:, i]],
                         labels=['$\\hat{y}_{' + str(i + 1) + '}$', '$y_{' + str(i + 1) + '}$'],
                         title='Output ' + str(i + 1) + ' prediction. MSE: ' + str(round(mse, 3)),
                         x_label='Time',
                         y_label='Output',
                         line_styles=['--', 'o-'],
                         markevery=markevery)

        # Save or show results
        now = datetime.datetime.now()
        if save_mode == 'all':
            now = datetime.datetime.now()
            plotter.save_pdf('results/' + results_and_models_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-exhaustion-test.pdf')
            plotter.save_eps('results/' + results_and_models_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-exhaustion-test')
        elif save_mode == 'pdf':
            plotter.save_pdf('results/' + results_and_models_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-exhaustion-test.pdf')
        elif save_mode == 'eps':
            plotter.save_eps('results/' + results_and_models_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-exhaustion-test')
        else:
            plotter.show()

        # Save model
        model.save('models/' + results_and_models_subdirectory + '/' +
                   now.strftime('%Y-%m-%d-%H-%M-%S') + '-exhausted-model')


class TTestContainer:
    def __init__(self):
        self.train_data = dict()
        self.np_val_X = None
        self.np_val_Y = None
        self.np_test_t = None
        self.np_test_X = None
        self.np_test_Y = None

    def check_key(self, T):
        if T not in self.train_data.keys():
            self.train_data[T] = dict()

    def get_train_u_X(self, T):
        return self.train_data[T]['np_train_u_X']

    def get_train_u_Y(self, T):
        return self.train_data[T]['np_train_u_Y']

    def get_train_f_X(self, T):
        return self.train_data[T]['np_train_f_X']

    def set_train_u_X(self, T, np_train_u_X):
        self.check_key(T)
        self.train_data[T]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, T, np_train_u_Y):
        self.check_key(T)
        self.train_data[T]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, T, np_train_f_X):
        self.check_key(T)
        self.train_data[T]['np_train_f_X'] = np_train_f_X


class NfNuTestContainer:
    def __init__(self):
        self.train_data = dict()
        self.np_val_X = None
        self.np_val_Y = None

    def check_key(self, nf, nu):
        if nf not in self.train_data.keys():
            self.train_data[nf] = dict()
            self.train_data[nf][nu] = dict()
        elif nu not in self.train_data[nf].keys():
            self.train_data[nf][nu] = dict()

    def get_train_u_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_X']

    def get_train_u_Y(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_Y']

    def get_train_f_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_f_X']

    def set_train_u_X(self, nf, nu, np_train_u_X):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, nf, nu, np_train_u_Y):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, nf, nu, np_train_f_X):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_f_X'] = np_train_f_X
