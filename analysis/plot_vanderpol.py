from util.data_container import ExhaustionTestContainer
from util.tests import ExhaustionTester
from util.plot import Plotter

tester = ExhaustionTester()

dictionary = tester.dao.load('../results/van_der_pol/2020-10-25-04-19-43-exhaustion-test/data.json')

data_container = ExhaustionTestContainer()
data_container.test_T = 0.5
data_container.load_results(dictionary)

## Plot
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

plotter = Plotter()
#tester.plot_graphs(data_container, plotter)

fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True

figsize=(5.4, 3.9)
fig, ax = plt.subplots(1, figsize=figsize)


loss_len = len(data_container.train_total_loss)
loss_x_axis = np.linspace(1, loss_len, loss_len)
# np_c_base = np.array([0, 255, 204]) / 255.0
# plotter.plot(x_axis=loss_x_axis,
#              y_axis_list=[np.array(data_container.train_total_loss), np.array(data_container.val_loss)],
#              labels=['Training', 'Validation'],
#              title='MSE',
#              x_label='Epoch',
#              y_label=None,
#              y_scale='log')

plotter.subplot(fig, ax, x_axis=loss_x_axis,
                y_axis_list=[np.array(data_container.train_total_loss), np.array(data_container.val_loss)],
                labels=['Training', 'Validation'],
                title=None,
                x_label='Epoch',
                y_scale='log',
                y_label='MSE',
                line_styles=['-', '-'],
                width=[-1, 1.5],
                colors=[[0.1, 0.5, 1], [0.9, 0.5, 1]]  # blue and pink
                #colors=['forestgreen', 'darkorange']
)
ax.axvline(x=6500, c='k', ls='--')

plt.tight_layout()

plotter.show()
