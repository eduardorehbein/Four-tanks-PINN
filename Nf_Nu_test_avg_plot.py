from util.data_container import NfNuTestContainer
from util.data_interface import JsonDAO
from util.plot import Plotter
import numpy as np

nfs_to_test = (2000, 4000, 10000, 100000)
nus_to_test = (40, 80, 100, 500, 1000)

paths = ['results/van_der_pol/2020-10-24-12-31-28-Nf-Nu-proportion-test/data.json',
         'results/van_der_pol/2020-10-28-09-52-26-Nf-Nu-proportion-test/data.json',
         'results/van_der_pol/2020-10-28-09-57-25-Nf-Nu-proportion-test/data.json',
         'results/van_der_pol/2020-10-28-10-10-01-Nf-Nu-proportion-test/data.json',
         'results/van_der_pol/2020-10-28-10-12-16-Nf-Nu-proportion-test/data.json']

final_val_losses_matrixes = list()
val_losses = {4000: {100: list()}, 10000: {80: list(), 500: list()}, 100000: {100: list(), 1000: list()}}
val_losses_len = 6350

dao = JsonDAO()

for path in paths:
    dictionary = dao.load(path)
    data_container = NfNuTestContainer()
    data_container.results = dictionary
    final_losses = data_container.get_final_val_losses(nfs_to_test, nus_to_test)
    final_val_losses_matrixes.append(final_losses)
    for nf in val_losses.keys():
        nf_dict = val_losses[nf]
        for nu in nf_dict.keys():
            val_losses[nf][nu].append(np.array(data_container.get_val_loss(nf, nu)[:val_losses_len]))

labels = list()
y_axis_list = list()
for nf in val_losses.keys():
    nf_dict = val_losses[nf]
    for nu in nf_dict.keys():
        y_axis_list.append(sum(val_losses[nf][nu])/len(val_losses[nf][nu]))
        labels.append(str(nf) + 'Nf and ' + str(nu) + 'Nu')

plot_matrix = sum(final_val_losses_matrixes)/len(final_val_losses_matrixes)

plotter = Plotter(fontsize=11)
figsize=(4.5, 3.2)

plotter.plot_heatmap(data=np.log10(plot_matrix),
                     title='L2 error', # validation
                     x_label='Nt',
                     y_label='Nf',
                     row_labels=nfs_to_test,
                     col_labels=nus_to_test,
                     figsize=figsize)

plotter.plot(x_axis=np.linspace(1, val_losses_len, val_losses_len),
             y_axis_list=y_axis_list,
             labels=labels,
             title='L2 error',
             x_label='Epoch',
             y_label=None,
             y_scale='log')
plotter.show()
