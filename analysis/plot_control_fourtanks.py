import numpy as np
#import tensorflow as tf
# from util.controller import PINNController
# from util.pinn import VanDerPolPINN
# from util.systems import VanDerPolSystem
# from util.plot import Plotter

# Constraints
np_min_v = np.array([[0.5, 0.5]])
np_max_v = np.array([[3.0, 3.0]])

np_min_h = np.array([[2.0, 2.0, 2.0, 2.0]])
np_max_h = np.array([[20.0, 20.0, 20.0, 20.0]])

# Inital condition
np_h0 = np_min_h

# Reference
np_ref = np.array([[9.159582958982018, 14.211590549777885, 2.1452569119256757, 3.7846331863579774],
                   [10.151255120853042, 15.750223780923415, 2.5530330191512176, 3.9140130625937135],
                   [8.226164209850278, 12.763340653079485, 2.385667521229082, 2.7180646268011888]])

# Controller and simulation parameters
T = 10.0
collocation_points_per_T = 10
prediction_horizon = 5*T
sim_time = 1200.0
outputs_to_control = [0, 1]

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a1': 0.071,  # [cm^2]
              'a2': 0.057,  # [cm^2]
              'a3': 0.071,  # [cm^2]
              'a4': 0.057,  # [cm^2]
              'A1': 28.0,  # [cm^2]
              'A2': 32.0,  # [cm^2]
              'A3': 28.0,  # [cm^2]
              'A4': 32.0,  # [cm^2]
              'alpha1': 0.5,  # [adm]
              'alpha2': 0.5,  # [adm]
              'k1': 3.33,  # [cm^3/Vs]
              'k2': 3.35,  # [cm^3/Vs]
              }

# Load model
# model = FourTanksPINN(sys_params=sys_params,
#                       hidden_layers=8,
#                       units_per_layer=20)
#model.load('models/four_tanks/2020-11-06-04-46-56-10dot0s-8l-20n-exhausted-model')
#model.trained_T = 10.0

## - load control results
from scipy.io import loadmat

d = loadmat("../results/fourtanks/control.mat")
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
figsize=(5.4, 5.1)
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize, sharex=True)

plotter.subplot(fig, ax1, x_axis=np_t,
                y_axis_list=[np_states[:, 0], np_states[:, 1], np_new_ref[:, 0], np_new_ref[:, 1]],  
                labels=['$h_1$', '$h_2$', None, None],
                title='Controlled tank levels',
                x_label=None,
                y_label='$h_1$,$h_2$ (cm)',
                line_styles=['-', '-', '--', '--'],
                width=[-1, 2.5, 1.3, 1.3],
                colors=['forestgreen', 'darkorange', 'k', [0.5]*3 ])
ax1.set_ylim((7,17))

plotter.subplot(fig, ax2, x_axis=np_t,
                y_axis_list=[np_states[:, 2], np_states[:, 3]],  
                labels=['$h_2$', '$h_3$'],
                title='Constrained tank levels', 
                x_label=None,
                y_label='$h_3$,$h_4$ (cm)',
                line_styles=['--', '--'],
                width=[-1, 2.5],
                colors=['forestgreen', 'darkorange'])

plotter.subplot(fig, ax3, x_axis=np_t,
                y_axis_list=[np_controls[:, 0], np_controls[:, 1]],
                labels=['$u_1$', '$u_2$'], 
                title=None, 
                x_label='Time (s)',
                y_label='pump voltage u (V)',
                draw_styles='steps',
                width=[1, 2],
                colors=[  np.array([88,126,245])/255., 'darkblue' ]
)

plt.tight_layout()
plotter.show()
