from util.data_container import StructTestContainer
from util.data_interface import JsonDAO
from util.plot import Plotter
import numpy as np

layers_to_test = (2, 4, 5, 8, 10)
neurons_per_layer_to_test = (3, 5, 10, 15, 20)

paths = ['results/van_der_pol/2020-10-11-09-02-45-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-07-07-43-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-07-48-25-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-07-58-41-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-08-50-27-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-08-56-52-nn-structural-test/data.json']

final_val_losses_matrixes = list()

dao = JsonDAO()

for path in paths:
    dictionary = dao.load(path)
    data_container = StructTestContainer()
    data_container.results = dictionary
    losses = data_container.get_final_val_losses(layers_to_test, neurons_per_layer_to_test)
    final_val_losses_matrixes.append(losses)

plot_matrix = sum(final_val_losses_matrixes)/len(final_val_losses_matrixes)

plotter = Plotter()
plotter.plot_heatmap(data=np.log10(plot_matrix),
                     title='Validation L2 error',
                     x_label='Neurons',
                     y_label='Layers',
                     row_labels=layers_to_test,
                     col_labels=neurons_per_layer_to_test)
plotter.show()
