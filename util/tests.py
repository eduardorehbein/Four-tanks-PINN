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

    def test(self, PINNModelClass, sys_params,
             np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
             results_subdirectory):
        # Normalizers
        X_normalizer = Normalizer()
        Y_normalizer = Normalizer()

        X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
        Y_normalizer.parametrize(np_train_u_Y)

        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('One tank neural network\'s structural test:' +
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
                model = PINNModelClass(sys_params=sys_params,
                                       hidden_layers=layers,
                                       units_per_layer=neurons,
                                       X_normalizer=X_normalizer,
                                       Y_normalizer=Y_normalizer)

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
    def __init__(self, df_time_var, df_scenario_var, df_X_var, df_Y_var, df_ic_var,
                 adam_epochs=500, max_lbfgs_iterations=1000):
        self.df_time_var = df_time_var
        self.df_scenario_var = df_scenario_var
        self.df_X_var = df_X_var
        self.df_Y_var = df_Y_var
        self.df_ic_var = df_ic_var

        self.adam_epochs = adam_epochs
        self.max_lbfgs_iterations = max_lbfgs_iterations

    def test(self, PINNModelClass, sys_params, hidden_layers, units_per_layer,
             train_dfs, test_df, train_scenarios, val_scenarios,
             results_subdirectory):
        # Plotter
        plotter = PdfPlotter()
        plotter.text_page('One tank neural network\'s working period test:' +
                          '\nAdam epochs -> ' + str(self.adam_epochs) +
                          '\nMax L-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
                          '\nTrain Nu -> ' + str(train_scenarios) +
                          '\nTrain Nf -> ' + str(train_dfs[0][train_dfs[0][self.df_scenario_var] == 1].shape[0] * train_scenarios) +
                          '\nValidation points -> ' + str(val_scenarios) +
                          '\nTest points -> ' + str(test_df.shape[0]))

        plot_dict = {'final train u losses': [], 'final train f losses': [],
                     'final train total losses': [], 'final val losses': [],
                     't': test_df[self.df_time_var].to_numpy(), 'y': test_df[self.df_Y_var].to_numpy(),
                     'nns': [], 'titles': []}

        tested_working_periods = []
        for df in train_dfs:
            # Current working period
            working_period = df[self.df_time_var].max()
            tested_working_periods.append(working_period)

            # Train data
            train_df = df[df[self.df_scenario_var] <= train_scenarios]
            train_u_df = train_df[train_df[self.df_time_var] == 0.0].sample(frac=1)
            np_train_u_X = train_u_df[self.df_X_var].to_numpy()
            np_train_u_Y = train_u_df[self.df_Y_var].to_numpy()
            np_train_f_X = train_df[self.df_X_var].sample(frac=1).to_numpy()

            # Validation data
            val_df = df[(df[self.df_scenario_var] > train_scenarios) &
                        (df[self.df_scenario_var] <= (train_scenarios + val_scenarios))].sample(frac=1)
            np_val_X = val_df[self.df_X_var].to_numpy()
            np_val_Y = val_df[self.df_Y_var].to_numpy()

            # Normalizers
            X_normalizer = Normalizer()
            Y_normalizer = Normalizer()

            X_normalizer.parametrize(np.concatenate([np_train_u_X, np_train_f_X]))
            Y_normalizer.parametrize(np_train_u_Y)

            # Instance PINN
            model = PINNModelClass(sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer)

            # Train
            print('Model training with working period of ' + str(working_period) + ' seconds:')
            model.train(np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
                        adam_epochs=self.adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

            # Test
            np_test_X = test_df[list(set(self.df_X_var) - set(self.df_ic_var))].to_numpy()
            test_ic = np.array(test_df[self.df_Y_var].to_numpy()[0])
            nn = model.predict(np_test_X, test_ic, working_period)
            plot_dict['nns'].append(nn)

            plot_dict['titles'].append('Working period = ' + str(round(working_period, 3)))

            # Final losses
            plot_dict['final train u losses'].append(model.train_u_loss[-1])
            plot_dict['final train f losses'].append(model.train_f_loss[-1])
            plot_dict['final train total losses'].append(model.train_total_loss[-1])
            plot_dict['final val losses'].append(model.validation_loss[-1])

        # Plot losses
        plotter.plot(x_axis=np.array(tested_working_periods),
                     y_axis_list=[np.array(plot_dict['final train total losses']),
                                  np.array(plot_dict['final val losses'])],
                     labels=['train loss', 'val loss'],
                     title='Train and validation total losses',
                     x_label='Working period [s]',
                     y_label='Loss [u²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(tested_working_periods),
                     y_axis_list=[np.array(plot_dict['final train u losses']),
                                  np.array(plot_dict['final train f losses'])],
                     labels=['u loss', 'f loss'],
                     title='Train losses',
                     x_label='Working period [s]',
                     y_label='Loss [u²]',
                     y_scale='log')

        # Plot test results
        for nn, title in zip(plot_dict['nns'], plot_dict['titles']):
            if len(nn.shape) == 1:
                mse = (np.square(plot_dict['y'] - nn)).mean()
                plotter.plot(x_axis=plot_dict['t'],
                             y_axis_list=[plot_dict['y'], nn],
                             labels=[self.df_Y_var[0], 'nn'],
                             title=title + ' Plot MSE: ' + str(round(mse, 3)) + ' u',
                             x_label='Time [s]',
                             y_label='Output [u]')
            else:
                transposed_nn = np.transpose(nn)
                transposed_y = np.transpose(plot_dict['y'])
                index = 0
                for current_nn, current_y in zip(transposed_nn, transposed_y):
                    y_label = self.df_Y_var[index]
                    index += 1
                    mse = (np.square(current_y - current_nn)).mean()
                    plotter.plot(x_axis=plot_dict['t'],
                                 y_axis_list=[current_y, current_nn],
                                 labels=[y_label, 'nn' + str(index)],
                                 title=title + ' Plot MSE: ' + str(round(mse, 3)) + ' u',
                                 x_label='Time [s]',
                                 y_label='Output [u]')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-working-period-test.pdf')


class NuNfTester:
    def __init__(self):
        pass

    def test(self):
        pass
