from util.plot import Plotter
import numpy as np

plotter = Plotter()

data = np.array([[1, 2, 0.1, 0.001, 0.02],
                [2, 3, 4, 0.05, 3]])
log_data = np.log10(data)

row_labels = [100, 1000]
col_labels = [2, 100, 1000, 10000, 100000]

plotter.plot_heatmap(log_data, 'Test title', row_labels, col_labels)

plotter.show()
