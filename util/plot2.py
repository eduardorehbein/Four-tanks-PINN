import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


fontsize = 11
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True


def plot_prediction_vdp(t, u, target, predictions,
                        x_scale='linear',
                        y_scale='linear', markevery=None, figsize=(4.5, 4)):

    x_label='Time (s)'
    title='Van der Pol'
    #labels=['$u$', '$\\hat{y}_1$', '$y_1$', '$\\hat{y}_2$', '$y_2$']
    line_styles=['-', '--', 'o-', '--', 'o-']
    draw_styles=['steps', 'default', 'default', 'default', 'default']
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    #plt.ylabel(y_label)
    #plt.xscale(x_scale)
    #plt.yscale(y_scale)

    linewidth = 3
    ax.plot(t, target[:,0], '-k', linewidth=linewidth)
    ax.plot(t, predictions[:,0], '-', color=[0.1, 0.5, 1],
            linewidth=linewidth, markevery='steps')
    
    ax2 = ax.twinx()
    c = [0.6]*3
    ax2.plot(t, u, color=c, linewidth=2.8, linestyle='dashed')

    for tl in ax2.get_yticklabels():
        tl.set_color(c)

    if x_scale == 'log':
        ax.set_xticks(t)
        ax.set_xticklabels( map(str, t) )

    #plt.legend()
    plt.tight_layout()
    plt.show()
