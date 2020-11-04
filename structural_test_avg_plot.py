from util.data_container import StructTestContainer
from util.data_interface import JsonDAO
from util.plot import Plotter
import numpy as np

layers_to_test = (2, 4, 5, 8, 10)
neurons_per_layer_to_test = (3, 5, 10, 15, 20)

paths = ['results/van_der_pol/2020-10-23-07-07-43-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-07-48-25-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-07-58-41-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-08-50-27-nn-structural-test/data.json',
         'results/van_der_pol/2020-10-23-08-56-52-nn-structural-test/data.json']

final_val_losses_matrixes = list()
val_losses = {4: list(), 5: list(), 8: list(), 10: list()}
val_losses_len = 6360

dao = JsonDAO()

for path in paths:
    dictionary = dao.load(path)
    data_container = StructTestContainer()
    data_container.results = dictionary
    final_losses = data_container.get_final_val_losses(layers_to_test, neurons_per_layer_to_test)
    final_val_losses_matrixes.append(final_losses)
    for layers in val_losses.keys():
        val_losses[layers].append(np.array(data_container.get_val_loss(layers, 20)[:val_losses_len]))

for layers in val_losses.keys():
    val_losses[layers] = sum(val_losses[layers])/len(val_losses[layers])

plot_matrix = sum(final_val_losses_matrixes)/len(final_val_losses_matrixes)

plotter = Plotter()
plotter.plot_heatmap(data=np.log10(plot_matrix),
                     title='Validation L2 error',
                     x_label='Neurons',
                     y_label='Layers',
                     row_labels=layers_to_test,
                     col_labels=neurons_per_layer_to_test)
loss_len = len(val_losses[4])
plotter.plot(x_axis=np.linspace(1, loss_len, loss_len),
             y_axis_list=list(val_losses.values()),
             labels=[str(layers) + 'L of 20N' for layers in val_losses.keys()],
             title='L2 error',
             x_label='Epoch',
             y_label=None,
             y_scale='log',
             line_styles=['-', '--', '-', '--'])
plotter.show()
