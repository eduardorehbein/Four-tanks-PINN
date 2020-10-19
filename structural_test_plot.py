from util.data_container import StructTestContainer
from util.tests import StructTester
from util.plot import Plotter
import matplotlib.pyplot as plt

tester = StructTester()
tester.layers_to_test = (2, 4, 5, 8, 10)
tester.neurons_per_layer_to_test = (3, 5, 10, 15, 20)

dictionary = tester.dao.load('results/van_der_pol/2020-10-11-09-02-45-nn-structural-test/data.json')

data_container = StructTestContainer()
data_container.results = dictionary

#plt.figure(figsize=(17.69, 3))
plotter = Plotter(fontsize=11)
graphs = tester.plot_graphs(data_container, plotter, just_val_loss=True, figsize=(4.5, 4))
plotter.show()
