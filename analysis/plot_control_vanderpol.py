import numpy as np
import tensorflow as tf
from util.controller import PINNController
from util.pinn import VanDerPolPINN
from util.systems import VanDerPolSystem
from util.plot import Plotter


# Constraints
np_min_u = np.array([[-1.0]])
np_max_u = np.array([[1.0]])

np_min_x = np.array([[-1.0, -1.0]])
np_max_x = np.array([[1.0, 1.0]])

# Initial condition
np_x0 = np.array([[0.0, 1.0]])

# Reference
np_ref = np.array([[0.0, 0.0],
                   [0.0, -0.5],
                   [0.0, 0.8],
                   [0.0, 1.0]])

# Controller and simulation parameters
T = 0.5
collocation_points_per_T = 10
prediction_horizon = 5*T
sim_time = 60.0

# Load model
# model = VanDerPolPINN(hidden_layers=4, units_per_layer=20)
# model.load('models/van_der_pol/2020-10-25-04-19-43-0dot5s-4l-20n-exhausted-model')

## - load control results
from scipy.io import savemat, loadmat
# savemat("../results/vanderpol/control.mat",
#         {'np_t': np_t, 'np_controls': np_controls,
#          'np_rk_controls': np_rk_controls,
#          'np_states': np_states,
#          'np_rk_states': np_rk_states,
#          'np_new_ref': np_new_ref,
#         }
# )

d = loadmat("../../results/vanderpol/control.mat")
np_t, np_controls, np_states, np_rk_controls, np_rk_states, np_new_ref = d['np_t'], d['np_controls'], d['np_states'], d['np_rk_controls'], d['np_rk_states'], d['np_new_ref']
np_t = np_t.T

# IAEs
pinn_iae = np.sum(np.abs(np_new_ref - np_states))
rk_iae = np.sum(np.abs(np_new_ref - np_rk_states))

print('PINN IAE:', pinn_iae)
print('Runge-Kutta IAE', rk_iae)

## Plot
import matplotlib.pyplot as plt

plotter = Plotter(fontsize=12)
#figsize=(3.4,5.2) # w=5.4, h=3.9)
figsize=(5.4,3.9) # w=5.4, h=3.9)
fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, sharex=True)

# Obs: Here we have switched x1 and x2 to plot according to the text notation
plotter.subplot(fig, ax1, x_axis=np_t,
                y_axis_list=[np_states[:, 1], np_states[:, 0], np_new_ref[:, 1]],  # np_rk_states[:, 1], np_rk_states[:, 0]
                labels=['$x_1$', '$x_2$', None],
                title=None, #'Van der Pol
                x_label=None,
                y_label='$\mathbf{x}$',
                line_styles=['-', '-', '--'],
                width=[-1, 2.5, 1.3],
                colors=[[0.1, 0.5, 1], [0.9, 0.5, 1], 'k'])
                #colors=['forestgreen', 'darkorange', 'k'])

plotter.subplot(fig, ax2, x_axis=np_t,
             y_axis_list=[np_controls[:, 0]], # np_rk_controls[:, 0]
             labels=None, #['PINN', 'Runge-Kutta'],
             title=None, #'Van der Pol control signal',
             x_label='Time (s)',
             y_label='u',
                draw_styles='steps',
                width=[1],
                colors=[  np.array([88,126,245])/255. ]
)

#plt.xlim((0, 10))
plt.tight_layout()
plotter.show()
