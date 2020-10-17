from util.data_container import TTestContainer
from util.tests import TTester
from util.plot import Plotter

tester = TTester()
tester.train_Ts = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)

dictionary = tester.dao.load('results/van_der_pol/2020-10-15-22-58-51-nn-T-test/data.json')

data_container = TTestContainer()
data_container.test_T = 0.5
data_container.load_results(dictionary)

plotter = Plotter()
tester.plot_graphs(data_container, plotter)
plotter.show()
