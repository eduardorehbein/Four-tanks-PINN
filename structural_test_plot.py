from util.data_container import StructTestContainer
from util.tests import StructTester
from util.plot import Plotter

# Tester
tester = StructTester()
tester.layers_to_test = (2, 4, 5, 8, 10)
tester.neurons_per_layer_to_test = (3, 5, 10, 15, 20)

# Loading data into a container
dictionary = tester.dao.load('results/van_der_pol/2020-10-23-08-56-52-nn-structural-test/data.json')
data_container = StructTestContainer()
data_container.results = dictionary

# Plot
plotter = Plotter()
tester.plot_graphs(data_container, plotter)
plotter.show()
