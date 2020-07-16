import numpy as np
import datetime
from util.normalizer import Normalizer
from util.plot import PdfPlotter


class StructTester:
    def __init__(self, layers_to_test, neurons_per_layer_to_test, max_adam_epochs=500, max_lbfgs_iterations=1000):
        self.layers_to_test = layers_to_test
        self.neurons_per_layer_to_test = neurons_per_layer_to_test
        self.max_adam_epochs = max_adam_epochs
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
                          '\nAdam epochs -> ' + str(self.max_adam_epochs) +
                          '\nL-BFGS iterations -> ' + str(self.max_lbfgs_iterations) +
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
                            max_adam_epochs=self.max_adam_epochs, max_lbfgs_iterations=self.max_lbfgs_iterations)

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
                     y_label='Loss [cm²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train total losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train total loss',
                     x_label='Neurons per layer',
                     y_label='Loss [cm²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train u losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train u loss',
                     x_label='Neurons per layer',
                     y_label='Loss [cm²]',
                     y_scale='log')
        plotter.plot(x_axis=np.array(self.neurons_per_layer_to_test),
                     y_axis_list=[np.array(plot_dict[layers]['final train f losses'])
                                  for layers in self.layers_to_test],
                     labels=[str(layers) + ' layers' for layers in self.layers_to_test],
                     title='Final train f loss',
                     x_label='Neurons per layer',
                     y_label='Loss [cm²]',
                     y_scale='log')

        # Save results
        now = datetime.datetime.now()
        plotter.save_pdf('results/' + results_subdirectory + '/' +
                         now.strftime('%Y-%m-%d-%H-%M-%S') + '-nn-structural-test.pdf')

