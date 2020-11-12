import os
import numpy as np
from datetime import datetime
from util.normalizer import Normalizer
from util.plot import Plotter
from util.data_interface import JsonDAO


class StructTester:
    def __init__(self, PINNModelClass=None, layers_to_test=None, neurons_per_layer_to_test=None,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None, random_seed=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.layers_to_test = layers_to_test
        self.neurons_per_layer_to_test = neurons_per_layer_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.random_seed = random_seed

        self.dao = JsonDAO()

    def test(self, data_container, results_subdirectory=None):
        # Start time
        start_time = datetime.now()

        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([data_container.np_train_u_X, data_container.np_train_f_X]))
        Y_normalizer.parametrize(data_container.np_train_u_Y)

        # Structural test
        for layers in self.layers_to_test:
            for neurons in self.neurons_per_layer_to_test:
                # Instance PINN
                if self.sys_params is None:
                    model = self.PINNModelClass(layers, neurons, X_normalizer, Y_normalizer, random_seed=None)
                else:
                    model = self.PINNModelClass(self.sys_params, layers, neurons, X_normalizer, Y_normalizer,
                                                random_seed=None)

                # Train
                print('Model training with ' + str(layers) + ' hidden layers of ' + str(neurons) + ' neurons:')
                model.train(data_container.np_train_u_X, data_container.np_train_u_Y, data_container.np_train_f_X,
                            data_container.train_T,
                            data_container.np_val_X, data_container.np_val_ic, data_container.val_T,
                            data_container.np_val_Y,
                            self.adam_epochs, self.max_lbfgs_iterations)

                # Save plot data
                data_container.set_train_u_loss(layers, neurons, model.train_u_loss)
                data_container.set_train_f_loss(layers, neurons, model.train_f_loss)
                data_container.set_train_total_loss(layers, neurons, model.train_total_loss)
                data_container.set_val_loss(layers, neurons, model.validation_loss)

        # Ending time
        data_container.test_duration = str(datetime.now() - start_time)

        # Plot results
        plotter = Plotter()
        plotter.text_page('Neural network\'s structural test:' +
                          '\nTest duration -> ' + data_container.test_duration +
                          '\nRandom seed -> ' + str(data_container.random_seed) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain T -> ' + str(data_container.train_T) + ' s' +
                          '\nTrain Nu -> ' + str(data_container.np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(data_container.np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(data_container.np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nPlot scale -> Log 10')
        self.plot_graphs(data_container, plotter)

        # Save or show results
        if results_subdirectory is not None:
            now = datetime.now()
            directory_path = 'results/' + results_subdirectory + '/' + now.strftime(
                '%Y-%m-%d-%H-%M-%S') + '-nn-structural-test'

            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

            plotter.save_pdf(directory_path + '/results.pdf')
            self.dao.save(directory_path + '/data.json', data_container.results)
        else:
            plotter.show()

    def plot_graphs(self, data_container, plotter, just_val_loss=False, figsize=(4.5, 4)):
        heatmap_colors = 'Greys' #'Reds'
        plotter.plot_heatmap(data=np.log10(data_container.get_final_val_losses(self.layers_to_test,
                                                                               self.neurons_per_layer_to_test)),
                             title='L2 error',
                             x_label='Number of Neurons',
                             y_label='Number of Layers',
                             row_labels=self.layers_to_test,
                             col_labels=self.neurons_per_layer_to_test,
                             imshow_kw={'cmap': heatmap_colors},
                             figsize=figsize
                             )
        if not just_val_loss:
            plotter.plot_heatmap(data=np.log10(data_container.
                                               get_final_train_total_losses(self.layers_to_test,
                                                                            self.neurons_per_layer_to_test)),
                                 title='Train L2 error',
                                 x_label='Neurons',
                                 y_label='Layers',
                                 row_labels=self.layers_to_test,
                                 col_labels=self.neurons_per_layer_to_test,
                                 imshow_kw={'cmap': heatmap_colors},
                                 figsize=figsize)
            plotter.plot_heatmap(data=np.log10(data_container.get_final_train_u_losses(self.layers_to_test,
                                                                                       self.neurons_per_layer_to_test)),
                                 title='Train u L2 error',
                                 x_label='Neurons',
                                 y_label='Layers',
                                 row_labels=self.layers_to_test,
                                 col_labels=self.neurons_per_layer_to_test,
                                 imshow_kw={'cmap': heatmap_colors},
                                 figsize=figsize)
            plotter.plot_heatmap(data=np.log10(data_container.get_final_train_f_losses(self.layers_to_test,
                                                                                       self.neurons_per_layer_to_test)),
                                 title='Train f L2 error',
                                 x_label='Neurons',
                                 y_label='Layers',
                                 row_labels=self.layers_to_test,
                                 col_labels=self.neurons_per_layer_to_test,
                                 imshow_kw={'cmap': heatmap_colors},
                                 figsize=figsize)



class TTester:
    def __init__(self, PINNModelClass=None, hidden_layers=None, units_per_layer=None, Ts=None,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None, random_seed=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.train_Ts = Ts

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.random_seed = random_seed

        self.dao = JsonDAO()

    def test(self, data_container, results_subdirectory=None):
        # Start time
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
                model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer,
                                            random_seed=self.random_seed)
            else:
                model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                            X_normalizer, Y_normalizer, random_seed=self.random_seed)

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
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nTrain Nu -> ' + str(nu) +
                          '\nTrain Nf -> ' + str(nf) +
                          '\nValidation points -> ' + str(val_points) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nTest points -> ' + str(test_points) +
                          '\nTest T -> ' + str(data_container.test_T) + ' s')
        self.plot_graphs(data_container, plotter)

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

    def plot_graph(self, data_container, plotter, type='validation error', color=[0, 153, 51], figsize=(4.5, 4)):
        import pdb
        #pdb.set_trace()
        np_train_Ts = np.array(self.train_Ts)
        np_c_base = np.array(color) / 255.0
        if type == 'validation error':
            plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_val_losses(self.train_Ts)],
                     labels=['val loss'],
                     title=None,
                     x_label='T',
                     y_label='L2 error',
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-',
                     np_c_base=np_c_base,
                     figsize =figsize)

    def plot_graphs(self, data_container, plotter):
        np_train_Ts = np.array(self.train_Ts)
        # np_c_base = np.array([0, 153, 51]) / 255.0
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_val_losses(self.train_Ts)],
                     labels=['val loss'],
                     title='Validation L2 error',
                     x_label='Train T',
                     y_label=None,
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-')
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_train_total_losses(self.train_Ts)],
                     labels=['train loss'],
                     title='Train total L2 error',
                     x_label='Train T',
                     y_label=None,
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-')
        plotter.plot(x_axis=np_train_Ts,
                     y_axis_list=[data_container.get_final_train_u_losses(self.train_Ts),
                                  data_container.get_final_train_f_losses(self.train_Ts)],
                     labels=['u', 'f'],
                     title='Train u and f L2 error',
                     x_label='Train T',
                     y_label=None,
                     x_scale='log',
                     y_scale='log',
                     line_styles='o-')

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
                             y_label=None,
                             line_styles=['--', 'o-'],
                             markevery=markevery)


class NfNuTester:
    def __init__(self, PINNModelClass=None, hidden_layers=None, units_per_layer=None, nfs_to_test=None, nus_to_test=None,
                 adam_epochs=500, max_lbfgs_iterations=2000, sys_params=None, random_seed=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.nfs_to_test = nfs_to_test
        self.nus_to_test = nus_to_test

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.random_seed = random_seed

        self.dao = JsonDAO()

    def test(self, data_container, results_subdirectory=None):
        # Start time
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
                    model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer,
                                                random_seed=self.random_seed)
                else:
                    model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                                X_normalizer, Y_normalizer, random_seed=self.random_seed)

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
                          '\nRandom seed -> ' + str(data_container.random_seed) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nTrain T -> ' + str(data_container.train_T) + ' s' +
                          '\nValidation points -> ' + str(data_container.np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nPlot scale -> Log 10')
        self.plot_graphs(data_container, plotter)

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

    def plot_graphs(self, data_container, plotter):
        # heatmap_colors = 'Oranges'
        plotter.plot_heatmap(data=np.log10(data_container.get_final_val_losses(self.nfs_to_test,
                                                                               self.nus_to_test)),
                             title='Validation L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_total_losses(self.nfs_to_test,
                                                                                       self.nus_to_test)),
                             title='Train total L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_u_losses(self.nfs_to_test,
                                                                                   self.nus_to_test)),
                             title='Train u L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)
        plotter.plot_heatmap(data=np.log10(data_container.get_final_train_f_losses(self.nfs_to_test,
                                                                                   self.nus_to_test)),
                             title='Train f L2 error',
                             x_label='Nu',
                             y_label='Nf',
                             row_labels=self.nfs_to_test,
                             col_labels=self.nus_to_test)


class ExhaustionTester:
    def __init__(self, PINNModelClass=None, hidden_layers=None, units_per_layer=None,
                 adam_epochs=500, max_lbfgs_iterations=10000, sys_params=None, random_seed=None):
        self.PINNModelClass = PINNModelClass
        self.sys_params = sys_params

        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

        self.random_seed = random_seed

        self.dao = JsonDAO()

    def test(self, data_container, results_and_models_subdirectory=None):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([data_container.np_train_u_X, data_container.np_train_f_X]))
        Y_normalizer.parametrize(data_container.np_train_u_Y)

        # Start time
        start_time = datetime.now()

        # Instance PINN
        if self.sys_params is None:
            model = self.PINNModelClass(self.hidden_layers, self.units_per_layer, X_normalizer, Y_normalizer,
                                        random_seed=self.random_seed)
        else:
            model = self.PINNModelClass(self.sys_params, self.hidden_layers, self.units_per_layer,
                                        X_normalizer, Y_normalizer, random_seed=self.random_seed)

        # Train
        model.train(data_container.np_train_u_X, data_container.np_train_u_Y, data_container.np_train_f_X,
                    data_container.train_T,
                    data_container.np_val_X, data_container.np_val_ic, data_container.val_T, data_container.np_val_Y,
                    self.adam_epochs, self.max_lbfgs_iterations)

        # Load train results into the container
        data_container.val_loss = model.validation_loss
        data_container.train_total_loss = model.train_total_loss
        data_container.train_u_loss = model.train_u_loss
        data_container.train_f_loss = model.train_f_loss

        # Test
        model_prediction = model.predict(data_container.np_test_X, data_container.np_test_ic, data_container.test_T)
        data_container.np_test_NN = model_prediction

        # Plotter
        plotter = Plotter()
        plotter.text_page('Exhaustion test:' +
                          '\nTest duration -> ' + str(datetime.now() - start_time) +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nNeural network\'s structure -> ' + str(self.hidden_layers) +
                          ' hidden layers of ' + str(self.units_per_layer) + ' neurons' +
                          '\nTrain T -> ' + str(data_container.train_T) + ' s' +
                          '\nTrain Nu -> ' + str(data_container.np_train_u_X.shape[0]) +
                          '\nTrain Nf -> ' + str(data_container.np_train_f_X.shape[0]) +
                          '\nValidation points -> ' + str(data_container.np_val_X.shape[0]) +
                          '\nValidation T -> ' + str(data_container.val_T) + ' s' +
                          '\nTest points -> ' + str(data_container.np_test_X.shape[0]) +
                          '\nTest T -> ' + str(data_container.test_T) + ' s',
                          vertical_position=0.25)
        self.plot_graphs(data_container, plotter)

        # Save model and results or show results
        if results_and_models_subdirectory is not None:
            # Save results
            now = datetime.now()
            directory_path = 'results/' + results_and_models_subdirectory + '/' + now.strftime(
                '%Y-%m-%d-%H-%M-%S') + '-exhaustion-test'

            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

            plotter.save_pdf(directory_path + '/results.pdf')
            self.dao.save(directory_path + '/data.json', data_container.get_results_dict())

            # Save model
            model_dir = 'models/' + results_and_models_subdirectory + '/' + now.strftime('%Y-%m-%d-%H-%M-%S') + '-' + \
                        str(data_container.train_T) + 's-' + str(self.hidden_layers) + 'l-' + \
                        str(self.units_per_layer) + 'n-exhausted-model'
            model_dir = model_dir.replace('.', 'dot')
            model.save(model_dir)
        else:
            plotter.show()

    def plot_graphs(self, data_container, plotter):
        # Plot train and validation losses
        loss_len = len(data_container.train_total_loss)
        loss_x_axis = np.linspace(1, loss_len, loss_len)
        # np_c_base = np.array([0, 255, 204]) / 255.0
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(data_container.train_total_loss), np.array(data_container.val_loss)],
                     labels=['Train', 'Validation'],
                     title='L2 error',
                     x_label='Epoch',
                     y_label=None,
                     y_scale='log')
        plotter.plot(x_axis=loss_x_axis,
                     y_axis_list=[np.array(data_container.train_u_loss), np.array(data_container.train_f_loss)],
                     labels=['u', 'f'],
                     title='Train L2 error',
                     x_label='Epoch',
                     y_label=None,
                     y_scale='log')

        # Plot test results
        np_test_U = data_container.get_np_test_U()
        plotter.plot(x_axis=data_container.np_test_t,
                     y_axis_list=[np_test_U[:, i] for i in range(np_test_U.shape[1])],
                     labels=['$u_{' + str(i + 1) + '}$' for i in range(np_test_U.shape[1])],
                     title='Input signal',
                     x_label='Time',
                     y_label=None,
                     draw_styles='steps')
        for i in range(data_container.np_test_Y.shape[1]):
            markevery = int(data_container.np_test_t.size / (data_container.np_test_t[-1] / data_container.test_T))
            mse = (np.square(data_container.np_test_NN[:, i] - data_container.np_test_Y[:, i])).mean()
            plotter.plot(x_axis=data_container.np_test_t,
                         y_axis_list=[data_container.np_test_Y[:, i], data_container.np_test_NN[:, i]],
                         labels=['$\\hat{y}_{' + str(i + 1) + '}$', '$y_{' + str(i + 1) + '}$'],
                         title='Output ' + str(i + 1) + ' prediction. MSE: ' + str(round(mse, 3)),
                         x_label='Time',
                         y_label=None,
                         line_styles=['--', 'o-'],
                         markevery=markevery)
