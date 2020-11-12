from util.data_container import NfNuTestContainer
from util.tests import NfNuTester
from util.plot import Plotter

tester = NfNuTester()
tester.nfs_to_test = (2000, 4000, 10000, 100000)
tester.nus_to_test = (40, 80, 100, 500, 1000)

dictionary = tester.dao.load('results/van_der_pol/2020-10-24-12-31-28-Nf-Nu-proportion-test/data.json')

data_container = NfNuTestContainer()
data_container.results = dictionary

plotter = Plotter()
tester.plot_graphs(data_container, plotter)
plotter.show()
