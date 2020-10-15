import os
import numpy as np
from datetime import datetime
from util.normalizer import Normalizer
from util.plot import Plotter
from util.data_interface import JsonDAO


class StructTester:
    def __init__(self, PINNModelClass, layers_to_test, neurons_per_layer_to_test,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.layers_to_test = layers_to_test
        self.neurons_per_layer_to_test = neurons_per_layer_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
             results_subdirectory=None, save_mode=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Plotter
        plotter = Plotter()
        plot_dict = dict()

        # Structural test
        start_time = datetime.now()
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
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
                            self.adam_epochs, self.max_lbfgs_iterations)

                # Save plot data
                plot_dict[layers]['final train u losses'].append(model.train_u_loss[-1])
                plot_dict[layers]['final train f losses'].append(model.train_f_loss[-1])
                plot_dict[layers]['final train total losses'].append(model.train_total_loss[-1])
                plot_dict[layers]['final val losses'].append(model.validation_loss[-1])

        # Plot results
        plotter.text_page('Neural network\'s structural test:' +
                          '\nTest duration -> ' + str(datetime.now() - start_time) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain T -> ' + str(train_T) + ' s' +
                          '\nTrain Nu -> ' + str(np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(val_T) + ' s' +
                          '\nPlot scale -> Log 10')

        heatmap_colors = 'Reds'
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final val losses'])
                                           for layers in self.layers_to_test])),
                             title='Validation L2 error',
                             x_label='Neurons',
                             y_label='Layers',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train total losses'])
                                           for layers in self.layers_to_test])),
                             title='Train total L2 error',
                             x_label='Neurons',
                             y_label='Layers',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train u losses'])
                                           for layers in self.layers_to_test])),
                             title='Train u L2 error',
                             x_label='Neurons',
                             y_label='Layers',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(np.array([np.array(plot_dict[layers]['final train f losses'])
                                           for layers in self.layers_to_test])),
                             title='Train f L2 error',
                             x_label='Neurons',
                             y_label='Layers',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test,
                             imshow_kw={'cmap': heatmap_colors})

        # Save or show results
        if save_mode == 'all':
            now = datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')
            plotter.save_eps('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test')
        elif save_mode == 'pdf':
            now = datetime.now()
            plotter.save_pdf('results/' + results_subdirectory + '/' +
                             now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')
        elif save_mode == 'eps':
            now = datetime.now()
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

        self.train_Ts = Ts

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.dao = JsonDAO()

    def test(self, data_container, results_subdirectory=None):
        start_time = datetime.now()
        for train_T in self.train_Ts:
            # Train data
            np_train_u_X = data_container.get_train_u_X(train_T)
            np_train_u_Y = data_container.get_train_u_Y(train_T)
            np_train_f_X = data_container.get_train_f_X(train_T)

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
            print('Model training with T of ' + str(train_T) + ' seconds:')
            model.train(np_train_u_X, np_train_u_Y, np_train_f_X, train_T,
                        data_container.np_val_X, data_container.np_val_ic, data_container.val_T,
                        data_container.np_val_Y,
                        adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

            # Test
            nn = model.predict(data_container.np_test_X, data_container.np_test_ic, data_container.test_T)

            data_container.set_nn(train_T, nn)
            data_container.set_title(train_T, 'T = ' + str(round(train_T, 3)) + ' s.')

            # Losses
            data_container.set_train_u_loss(train_T, model.train_u_loss)
            data_container.set_train_f_loss(train_T, model.train_f_loss)
            data_container.set_train_total_loss(train_T, model.train_total_loss)
            data_container.set_val_loss(train_T, model.validation_loss)

        # Plot front page and losses
        nu = data_container.get_train_u_X(self.train_Ts[0]).shape[0]
        nf = data_container.get_train_f_X(self.train_Ts[0]).shape[0]
        val_points = data_container.np_val_X.shape[0]
        test_points = data_container.np_test_X.shape[0]

        plotter = Plotter()
        plotter.text_page('Neural network\'s T test:' +
                          '\nTest duration -> ' + str(datetime.now() - start_time) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain Nu -> ' + str(nu) +
                          '\nTrain Nf -> ' + str(nf) +
                          '\nValidation points -> ' + str(val_points) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nTest points -> ' + str(test_points) +
                          '\nTest T -> ' + str(data_container.test_T) + ' s')

        np_train_Ts = np.array(self.train_Ts)
        np_c_base = np.array([0, 153, 51]) / 255.0
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_val_losses(self.train_Ts)],
                     labels=['val loss'],
                     title='Validation loss',
                     x_label='Train T',
                     y_label='Loss',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-',
                     np_c_base=np_c_base)
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_train_total_losses(self.train_Ts)],
                     labels=['train loss'],
                     title='Train total loss',
                     x_label='Train T',
                     y_label='Loss',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-',
                     np_c_base=np_c_base)
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_train_u_losses(self.train_Ts),
                                  data_container.get_final_train_f_losses(self.train_Ts)],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='Train T',
                     y_label='Loss',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-',
                     np_c_base=np_c_base)

        # Plot test results
        for nn, title, current_T in zip(data_container.get_nns(self.train_Ts),
                                        data_container.get_titles(self.train_Ts),
                                        np_train_Ts):
            transposed_nn = np.transpose(nn)
            transposed_y = np.transpose(data_container.np_test_Y)
            markevery = int(data_container.np_test_t.size / (data_container.np_test_t[-1] / data_container.test_T))
            output_index = 0
            for current_nn, current_y in zip(transposed_nn, transposed_y):
                output_index += 1
                mse = (np.square(current_y - current_nn)).mean()
                plotter.plot(x_axis=data_container.np_test_t,
                             y_axis_list=[current_y, current_nn],
                             labels=['$\\hat{y}_{' + str(output_index) + '}$', '$y_{' + str(output_index) + '}$'],
                             title=title + ' MSE: ' + str(round(mse, 3)),
                             x_label='Time',
                             y_label='Output',
                             line_styles=['--', 'o-'],
                             markevery=markevery,
                             np_c_base=np_c_base)

        # Save or show results
        if results_subdirectory is not None:
            now = datetime.now()
            directory_path = 'results/' + results_subdirectory + '/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-T-test'

            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

            plotter.save_pdf(directory_path + '/results.pdf')
            self.dao.save(directory_path + '/data.json', data_container.get_results_dict())
        else:
            plotter.show()


class NfNuTester:
    def __init__(self, PINNModelClass, hidden_layers, units_per_layer, nfs_to_test, nus_to_test,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.nfs_to_test = nfs_to_test
        self.nus_to_test = nus_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.dao = JsonDAO()

    def test(self, data_container, results_subdirectory=None):
        start_time = datetime.now()
        for nf in self.nfs_to_test:
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
                model.train(np_train_u_X, np_train_u_Y, np_train_f_X, data_container.train_T,
                            data_container.np_val_X, data_container.np_val_ic, data_container.val_T,
                            data_container.np_val_Y,
                            adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

                # Save plot data
                data_container.set_train_u_loss(nf, nu, model.train_u_loss)
                data_container.set_train_f_loss(nf, nu, model.train_f_loss)
                data_container.set_train_total_loss(nf, nu, model.train_total_loss)
                data_container.set_val_loss(nf, nu, model.validation_loss)

        # Plot results
        plotter = Plotter()
        plotter.text_page('Neural network\'s Nf/Nu test:' +
                          '\nTest duration -> ' + str(datetime.now() - start_time) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nTrain T -> ' + str(data_container.train_T) + ' s' +
                          '\nValidation points -> ' + str(data_container.np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nPlot scale -> Log 10')

        heatmap_colors = 'Oranges'
        plotter.plot_heatmap(data=np.log10(data_container.get_final_val_losses(self.nfs_to_test,
                                                                               self.nus_to_test)),
                             title='Validation L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_total_losses(self.nfs_to_test,
                                                                                       self.nus_to_test)),
                             title='Train total L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_u_losses(self.nfs_to_test,
                                                                                   self.nus_to_test)),
                             title='Train u L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test,
                             imshow_kw={'cmap': heatmap_colors})
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_f_losses(self.nfs_to_test,
                                                                                   self.nus_to_test)),
                             title='Train f L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test,
                             imshow_kw={'cmap': heatmap_colors})

        # Save or show results
        if results_subdirectory is not None:
            now = datetime.now()
            directory_path = 'results/' + results_subdirectory + '/' + now.strftime(
                '%Y-%m-%d-%H-%M-%S') + '-Nf-Nu-proportion-test'

            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

            plotter.save_pdf(directory_path + '/results.pdf')
            self.dao.save(directory_path + '/data.json', data_container.results)
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

    def test(self, np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
             np_test_t, np_test_X, np_test_ic, test_T, np_test_Y, results_and_models_subdirectory, save_mode=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Start time
        start_time = datetime.now()

        # Instance PINN
        if self.sys_params is None:
            model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer)
        else:
            model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                        X_normalizer, Y_normalizer)

        # Train
        model.train(np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
                    self.adam_epochs, self.max_lbfgs_iterations)

        # Calculate controls signals an their T
        np_test_U = self.get_test_control(np_test_t, np_test_X)

        # Test
        model_prediction = model.predict(np_test_X, np_test_ic, test_T)

        # Plotter
        plotter = Plotter()
        plotter.text_page('Exhaustion test:' +
                          '\nTest duration -> ' + str(datetime.now() - start_time) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nTrain T -> ' + str(train_T) + ' s' +
                          '\nTrain Nu -> ' + str(np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(val_T) + ' s' +
                          '\nTest points -> ' + str(np_test_X.shape[0]) +
                          '\nTest T -> ' + str(test_T) + ' s',
                          vertical_position=0.25)

        # Plot train and validation losses
        loss_len = len(model.train_total_loss)
        loss_x_axis = np.linspace(1, loss_len, loss_len)
        np_c_base = np.array([0, 255, 204])/255.0
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(model.train_total_loss), np.array(model.validation_loss)],
                     labels=['Train loss', 'Validation loss'],
                     title='Total losses',
                     x_label='Epoch',
                     y_label='Loss',
                     y_scale='log',
                     np_c_base=np_c_base)
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(model.train_u_loss), np.array(model.train_f_loss)],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='Epoch',
                     y_label='Loss',
                     y_scale='log',
                     np_c_base=np_c_base)

        # Plot test results
        plotter.plot(x_axis=np_test_t,
                     y_axis_list=[np_test_U[:, i] for i in range(np_test_U.shape[1])],
                     labels=['$u_{' + str(i + 1) + '}$' for i in range(np_test_U.shape[1])],
                     title='Input signal',
                     x_label='Time',
                     y_label='Input',
                     draw_styles='steps',
                     np_c_base=np_c_base)
        for i in range(np_test_Y.shape[1]):
            markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
            mse = (np.square(model_prediction[:, i] - np_test_Y[:, i])).mean()
            plotter.plot(x_axis=np_test_t,
                         y_axis_list=[np_test_Y[:, i], model_prediction[:, i]],
                         labels=['$\\hat{y}_{' + str(i + 1) + '}$', '$y_{' + str(i + 1) + '}$'],
                         title='Output ' + str(i + 1) + ' prediction. MSE: ' + str(round(mse, 3)),
                         x_label='Time',
                         y_label='Output',
                         line_styles=['--', 'o-'],
                         markevery=markevery,
                         np_c_base=np_c_base)

        # Save or show results
        now = datetime.now()
        if save_mode == 'all':
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

    def get_test_control(self, np_test_t, np_test_X):
        t_index = 0
        while not np.array_equal(np_test_X[:, t_index].flatten(), np_test_t.flatten()):
            t_index = t_index + 1
        return np.delete(np_test_X, t_index, axis=1)
