import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PdfPlotter:
    def __init__(self):
        self.y_range = float('-inf')

    def plot(self, x_axis, y_axis_list, labels, title, x_label, y_label, limit_range=False, y_scale='linear'):
        if len(y_axis_list) != len(labels):
            raise Exception('x_axis_list\'s length and label\'s length do not match.')
        else:
            plt.figure()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.yscale(y_scale)
            y_min = float('inf')
            for i in range(len(y_axis_list)):
                np_y = y_axis_list[i]
                np_y_min = np_y.min()
                if np_y_min < y_min:
                    y_min = np_y_min
                plt.plot(x_axis, np_y, label=labels[i])
            if limit_range:
                axes = plt.gca()
                axes.set_ylim([y_min, y_min + self.y_range])
            plt.legend()

    def set_y_range(self, y_axis_list):
        min_min = float('inf')
        max_max = float('-inf')
        for np_y in y_axis_list:
            min_min = min(min_min, np_y.min())
            max_max = max(max_max, np_y.max())
        data_range = abs(max_max - min_min)
        if data_range > self.y_range:
            self.y_range = data_range

    def save_pdf(self, path):
        pdf = PdfPages(path)
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()
