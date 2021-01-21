import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    """Cartesian and heatmap plots"""

    def __init__(self, font_family='serif', font='Times New Roman', mathtext_font='stix'):
        """
        Font setup.

        :param font_family: Font family
            (default is 'serif')
        :type font_family: str
        :param font: Font
            (default is 'Times New Roman')
        :type font: str
        :param mathtext_font: Math text font
            (default is 'stix')
        :type mathtext_font: str
        """

        rcParams['font.family'] = font_family
        rcParams['font.sans-serif'] = [font]
        rcParams['mathtext.fontset'] = mathtext_font

    def text_page(self, text, vertical_position=0.3, size=24):
        """
        It plots a text page.

        :param text: Text to be plotted
        :type text: str
        :param vertical_position: Vertical position of the text
            (default is 0.3)
        :type vertical_position: int or float
        :param size: Text size
            (default is 24)
        :type size: int of float
        """

        firstPage = plt.figure(figsize=(11.69, 8.27))
        firstPage.clf()
        firstPage.text(0.5, vertical_position, text, transform=firstPage.transFigure, size=size, ha="center")

    def plot(self, x_axis, y_axis_list, labels, title, x_label, y_label,
             x_scale='linear', y_scale='linear', line_styles='-', markevery=None, draw_styles='default',
             np_c_base=np.array([200, 200, 200])/255):
        """
        It plots one or multiple curves in a single graph.

        :param x_axis: X axis values
        :type x_axis: numpy.ndarray
        :param y_axis_list: List with Y axis values vectors. Each vector is plotted in a different curve
        :type y_axis_list: list
        :param labels: Labels for each curve
        :type labels: list
        :param title: Plot title
        :type title: str
        :param x_label: X axis label
        :type x_label: str or None
        :param y_label: Y axis label
        :type y_label: str or None
        :param x_scale: X axis scale. Examples: 'linear', 'log10', etc
            (default is 'linear')
        :type x_scale: str
        :param y_scale: Y axis scale. Examples: 'linear', 'log10', etc
            (default is 'linear')
        :type y_scale: str
        :param line_styles: Line styles. Examples: '-', 'o-', etc
            (default is '-')
        :type line_styles: str or list
        :param markevery: Mark every n samples
            (default is None)
        :type markevery: int
        :param draw_styles: Styles of drawing between samples. Examples: 'default', 'steps', etc
            (default is 'default')
        :type draw_styles: str or list
        :param np_c_base: RGB color base
            (default is np.array([200, 200, 200])/255)
        :type np_c_base: numpy.ndarray or None
        """

        if len(y_axis_list) != len(labels):
            raise Exception('y_axis_list length and label length do not match.')
        else:
            plt.figure()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xscale(x_scale)
            plt.yscale(y_scale)

            # Color step for multiple curve plotting
            c_step = 1/len(y_axis_list)

            # Plotting of each curve
            for i in range(len(y_axis_list)):
                # Curve selection
                np_y = y_axis_list[i]

                # Curve style
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

                # Plot
                plt.plot(x_axis, np_y, line_style,
                         label=labels[i], c=color, markevery=markevery, ds=ds)

            # Legend
            if len(y_axis_list) > 1:
                plt.legend()

    def multiplot(self, x_axis, y_axis_matrices, labels_list, title, x_label, y_labels_list,
                  line_styles='-', markevery=None, draw_styles='default',
                  np_c_base=np.array([200, 200, 200])/255):
        """
        It plots one or multiple curves in multiple graphs that share the same X axis.

        :param x_axis: X axis values
        :type x_axis: numpy.ndarray
        :param y_axis_matrices: List of list of Y axis values vectors. Each vector is plotted in a different curve in a
            graph, while each list of vectors is plotted in a different graph.
        :type y_axis_matrices: list
        :param labels_list: List of list of labels. Same idea presented in y_axis_matrices
        :type labels_list: list
        :param title: Multiplot title
        :type title: str
        :param x_label: X axis label
        :type x_label: str
        :param y_labels_list: List of Y axis labels
        :type y_labels_list: list
        :param line_styles: Line style for each curve. Same idea presented in y_axis_matrices
            (default is '-')
        :type line_styles: str or list
        :param markevery: Mark every n samples
            (default is None)
        :type markevery: int
        :param draw_styles: Styles of drawing between samples. Same idea presented in y_axis_matrices
            (default is 'default')
        :type draw_styles: str or list
        :param np_c_base: RGB color base
            (default is np.array([200, 200, 200])/255)
        :type np_c_base: numpy.ndarray or None
        """

        # Figure and graphs
        rows = len(y_axis_matrices)
        fig, axs = plt.subplots(rows, sharex=True)

        # Title
        fig.suptitle(title)

        # Plotting in each graph
        for i, (ax, y_axis_list, labels, y_label) \
                in enumerate(zip(axs, y_axis_matrices, labels_list, y_labels_list)):
            if len(y_axis_list) != len(labels):
                raise Exception('y_axis_list length and label length do not match.')
            else:
                # Y label
                ax.set(ylabel=y_label)

                # Color step for multiple curve plotting
                c_step = 1/len(y_axis_list)

                # Plotting of each curve
                for j in range(len(y_axis_list)):
                    # Curve selection
                    np_y = y_axis_list[j]

                    # Curve style
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

                    # Plot
                    ax.plot(x_axis, np_y, line_style,
                            label=labels[j], c=color, markevery=markevery, ds=ds)

                # Legend
                if len(y_axis_list) > 1:
                    ax.legend()
        # X label
        axs[-1].set(xlabel=x_label)

    def plot_heatmap(self, data, title, x_label, y_label, row_labels, col_labels,
                     cbar_kw={}, imshow_kw={'cmap': 'Greys'}, txt_val_fmt='{x:.2f}', txt_colors=('black', 'white'),
                     txt_threshold=None, text_kw={}):
        """
        It plots a table like heatmap.

        :param data: Data
        :type data: numpy.ndarray
        :param title: Title
        :type title: str
        :param x_label: X axis label
        :type x_label: str
        :param y_label: Y axis label
        :type y_label: str
        :param row_labels: Row labels
        :type row_labels: list or tuple
        :param col_labels: Column labels
        :type col_labels: list or tuple
        :param cbar_kw: Color bar args
            (default is {})
        :type cbar_kw: dict
        :param imshow_kw: Imshow args
            (default is {'cmap': 'Greys'}})
        :type imshow_kw: dict
        :param txt_val_fmt: Format of the annotations inside the heatmap
            (default is '{x:.2f}')
        :type txt_val_fmt: str
        :param txt_colors: Text colors
            (default is ('black', 'white'))
        :type txt_colors: tuple or list
        :param txt_threshold: Value in data units according to which the colors from text colors are applied
            (default is None)
        :type txt_threshold: float
        :param text_kw: Extra args for the heatmap annotation
            (default is {})
        :type text_kw: dict
        """

        fig, ax = plt.subplots()
        im, cbar = self.get_heatmap(data, title, x_label, y_label, row_labels, col_labels,
                                    ax=ax, cbar_kw=cbar_kw, **imshow_kw)
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

        # Heatmap plot
        im = ax.imshow(data, **kwargs)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Rotating the tick labels and setting their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Turning spines off and creating white grid.
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
            Data used to annotate.  If None, the image data is used.  Optional.
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

        # Normalizing the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Setting default alignment to center, but allowing it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Getting the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = StrMethodFormatter(valfmt)

        # Looping over the data and creating a `Text` for each "pixel".
        # Changing the text color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def save_pdf(self, path):
        """
        It saves all figures in a PDF.

        :param path: File path
        :type path: str
        """

        pdf = PdfPages(path)
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()

    def show(self):
        """It shows all figures"""

        plt.show()
