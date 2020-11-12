import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    def __init__(self, font_family='serif', font='Times New Roman', mathtext_font='stix', fontsize=12):
        rcParams['font.family'] = font_family
        rcParams['font.sans-serif'] = [font]
        rcParams['mathtext.fontset'] = mathtext_font
        rcParams['axes.titlesize'] = fontsize
        rcParams['axes.labelsize'] = fontsize

    def text_page(self, text, vertical_position=0.3, size=24):
        firstPage = plt.figure(figsize=(21.69, 8.27))
        firstPage.clf()
        firstPage.text(0.5, vertical_position, text, transform=firstPage.transFigure, size=size, ha="center")

    def plot(self, x_axis, y_axis_list, labels, title, x_label, y_label,
             x_scale='linear', y_scale='linear', line_styles='-',
             markevery=None, draw_styles='default', np_c_base=np.array([200, 200, 200])/255, figsize=(4.5, 4)):
        if len(y_axis_list) != len(labels):
            raise Exception('y_axis_list\'s length and label\'s length do not match.')
        else:
            #plt.figure()
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xscale(x_scale)
            plt.yscale(y_scale)

            c_step = 1/len(y_axis_list)
            for i in range(len(y_axis_list)):
                np_y = y_axis_list[i]
                if isinstance(line_styles, str):
                    line_style = line_styles
                else:
                    line_style = line_styles[i]
                if isinstance(draw_styles, str):
                    ds = draw_styles
                else:
                    ds = draw_styles[i]
                if np_c_base is None:
                    color = None
                else:
                    color = c_step * (i + 1) * np_c_base
                plt.plot(x_axis, np_y, line_style,
                         label=labels[i], c=color, markevery=markevery, ds=ds)
            if x_scale == 'log':
                ax.set_xticks(x_axis)
                ax.set_xticklabels( map(str, x_axis) )

            if len(y_axis_list) > 1:
                plt.legend()

            plt.tight_layout()

    def multiplot(self, x_axis, y_axis_matrices, labels_list, title, x_label, y_labels_list,
                  line_styles='-', markevery=None, draw_styles='default',
                  np_c_base=np.array([200, 200, 200])/255):
        rows = len(y_axis_matrices)
        fig, axs = plt.subplots(rows, sharex=True)
        fig.suptitle(title)
        for i, (ax, y_axis_list, labels, y_label) \
                in enumerate(zip(axs, y_axis_matrices, labels_list, y_labels_list)):
            if len(y_axis_list) != len(labels):
                raise Exception('y_axis_list\'s length and label\'s length do not match.')
            else:
                ax.set(ylabel=y_label)
                c_step = 1/len(y_axis_list)
                for j in range(len(y_axis_list)):
                    np_y = y_axis_list[j]
                    if isinstance(line_styles, str):
                        line_style = line_styles
                    else:
                        line_style = line_styles[i][j]
                    if isinstance(draw_styles, str):
                        ds = draw_styles
                    else:
                        ds = draw_styles[i][j]
                    if np_c_base is None:
                        color = None
                    else:
                        color = c_step * (j + 1) * np_c_base
                    ax.plot(x_axis, np_y, line_style,
                            label=labels[j], c=color, markevery=markevery, ds=ds)
                if len(y_axis_list) > 1:
                    ax.legend()
        axs[-1].set(xlabel=x_label)

    def plot_heatmap(self, data, title, x_label, y_label, row_labels, col_labels,
                     cbar_kw={}, imshow_kw={'cmap': 'Greys'}, txt_val_fmt="{x:.2f}", txt_colors=("black", "white"),
                     txt_threshold=None, figsize=(4.5, 4), text_kw={}):
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = self.get_heatmap(data, title, x_label, y_label, row_labels, col_labels, ax=ax, cbar_kw=cbar_kw,
                                   **imshow_kw)
        im, cbar = heatmap
        self.annotate_heatmap(im, data, txt_val_fmt, txt_colors, txt_threshold, **text_kw)

    def get_heatmap(self, data, title, x_label, y_label, row_labels, col_labels, ax=None, cbar_kw={}, **kwargs):
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
        #import pdb
        #pdb.set_trace()
        im = ax.imshow(data, **kwargs)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # ERIC: I disabled this for heatmap since not needed

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()

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

    def show(self):
        plt.show()
