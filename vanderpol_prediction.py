import numpy as np
import tensorflow as tf
import pandas as pd
from util.pinn import VanDerPolPINN
from util.plot2 import *

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Instance PINN
model = VanDerPolPINN(hidden_layers=4, units_per_layer=20)
model.trained_T = 0.5

# Load model
model.load('models/van_der_pol/2020-10-25-04-19-43-0dot5s-4l-20n-exhausted-model')

# Test data
test_df = pd.read_csv('data/van_der_pol/rand_seed_10_sim_time_10.0s_200_collocation_points.csv')

np_test_t = test_df['t'].to_numpy()
np_test_u = test_df['u'].to_numpy()
np_test_X = test_df[['t', 'u']].to_numpy()
np_test_y = test_df[['x1', 'x2']].to_numpy()
np_test_ic = np.reshape(np_test_y[0, :], (1, np_test_y.shape[1]))

# Model prediction
test_T = 0.5
np_test_nn = model.predict(np_test_X, np_test_ic, prediction_T=test_T)

## - Plot test results
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True

markevery = int(np_test_t.size / (np_test_t[-1] / test_T))
mse = (np.square(np_test_nn - np_test_y)).mean()

#plot_prediction_vdp(t=np_test_t, u=np_test_u, target=np_test_y, predictions=np_test_nn, markevery=markevery)
             
t=np_test_t
u=np_test_u
target=np_test_y
predictions=np_test_nn
figsize=(5.5, 3.4)
x_scale='linear'
y_scale='linear'
                        
x_label='Time (s)'
title='Van der Pol prediction'
#labels=['$u$', '$\\hat{y}_1$', '$y_1$', '$\\hat{y}_2$', '$y_2$']
line_styles=['-', '--', 'o-', '--', 'o-']
draw_styles=['steps', 'default', 'default', 'default', 'default']
    
fig, ax = plt.subplots(figsize=figsize)
plt.title(title)
plt.xlabel(x_label)
#plt.ylabel(y_label)
#plt.xscale(x_scale)
#plt.yscale(y_scale)

linewidth = 1
ax.plot(t, target[:,0], '-k', linewidth=linewidth)
ax.plot(t, predictions[:,0], 'o-', color=[0.1, 0.5, 1],  # blue, x1
        linewidth=linewidth*1.2, markevery=markevery)

ax.plot(t, target[:,1], '-', linewidth=linewidth)
ax.plot(t, predictions[:,1], 'o-', color=[1, 0.5, 1],  # pinc, x2
        linewidth=linewidth*1.2, markevery=markevery)

ax.set_ylabel('outputs $y_1$, $y_2$', fontsize=fontsize)

c = [0.6]*3

ax2 = ax.twinx()
ax2.set_ylabel('input $u$', color=c)
ax2.plot(t, u, color=c, linewidth=2.3, linestyle='dashed', ds='steps')

for tl in ax2.get_yticklabels():
    tl.set_color( [0.5]*3 )

if x_scale == 'log':
    ax.set_xticks(t)
    ax.set_xticklabels( map(str, t) )

#plt.legend()
plt.tight_layout()
plt.show()
