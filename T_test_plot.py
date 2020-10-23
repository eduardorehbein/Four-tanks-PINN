from util.data_container import TTestContainer
from util.tests import TTester
from util.plot import Plotter
import numpy as np

tester = TTester()
tester.train_Ts = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)

dictionary = tester.dao.load('results/van_der_pol/2020-10-15-22-58-51-nn-T-test/data.json')

data_container = TTestContainer()
data_container.test_T = 0.5
data_container.load_results(dictionary)

plotter = Plotter()

loss_05 = data_container.get_val_loss(0.5)
loss_1 = data_container.get_val_loss(1.0)

x_len = min(len(loss_05), len(loss_1))
plotter.plot(x_axis=np.linspace(1, x_len, x_len),
             y_axis_list=[np.array(loss_05[:x_len]), np.array(loss_1[:x_len])],
             labels=['T = 0.5s', 'T = 1.0s'],
             title='L2 error',
             x_label='Epoch',
             y_label=None,
             y_scale='log')
# tester.plot_graphs(data_container, plotter)
plotter.show()
