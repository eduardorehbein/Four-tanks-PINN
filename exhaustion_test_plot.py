from util.data_container import ExhaustionTestContainer
from util.tests import ExhaustionTester
from util.plot import Plotter

tester = ExhaustionTester()

dictionary = tester.dao.load('results/van_der_pol/2020-10-19-12-37-58-exhaustion-test/data.json')

data_container = ExhaustionTestContainer()
data_container.test_T = 0.5
data_container.load_results(dictionary)

plotter = Plotter()
tester.plot_graphs(data_container, plotter)
plotter.show()
