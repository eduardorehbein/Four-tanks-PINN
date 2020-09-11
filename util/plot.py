import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter


class Plotter:
    def __init__(self, font_family='serif', font='Times New Roman'):
        self.y_range = float('-inf')
        rcParams['font.family'] = font_family
        rcParams['font.sans-serif'] = [font]

    def text_page(self, text, vertical_position=0.4, size=24):
        firstPage = plt.figure(figsize=(11.69, 8.27))
        firstPage.clf()
        firstPage.text(0.5, vertical_position, text, transform=firstPage.transFigure, size=size, ha="center")

    def plot(self, x_axis, y_axis_list, labels, title, x_label, y_label,
             limit_range=False, x_scale='linear', y_scale='linear', line_style='-'):
        if len(y_axis_list) != len(labels):
            raise Exception('y_axis_list\'s length and label\'s length do not match.')
        else:
            plt.figure()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xscale(x_scale)
            plt.yscale(y_scale)

            c_step = 1/len(y_axis_list)
            y_min = float('inf')
            for i in range(len(y_axis_list)):
                np_y = y_axis_list[i]
                np_y_min = np_y.min()
                if np_y_min < y_min:
                    y_min = np_y_min
                plt.plot(x_axis, np_y, line_style, label=labels[i], c=str(c_step * i))
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

    def plot_heatmap(self, data, title, row_labels, col_labels,
                     cbar_kw={}, cbar_label="L2 error", imshow_kw={'cmap': 'Greys'},
                     txt_val_fmt="{x:.2f}", txt_colors=("black", "white"), txt_threshold=None, text_kw={}):
        fig, ax = plt.subplots()
        im, cbar = self.get_heatmap(data, row_labels, col_labels,
                                    ax=ax, cbar_kw=cbar_kw, cbarlabel=cbar_label, **imshow_kw)
        self.annotate_heatmap(im, data, txt_val_fmt, txt_colors, txt_threshold, **text_kw)
        plt.title(title)

    def get_heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def save_pdf(self, path):
        pdf = PdfPages(path)
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()

    def save_eps(self, directory_path):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        for fig in range(1, plt.gcf().number + 1):
            plt.figure(fig)
            plt.savefig(directory_path + '/figure_' + str(fig) + '.eps', format='eps')

    def show(self):
        plt.show()
