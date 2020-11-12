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

d = loadmat("../results/vanderpol/control.mat")
np_t, np_controls, np_states, np_rk_controls, np_rk_states, np_new_ref = d['np_t'], d['np_controls'], d['np_states'], d['np_rk_controls'], d['np_rk_states'], d['np_new_ref']
np_t = np_t.T

# IAEs
pinn_iae = np.sum(np.abs(np_new_ref - np_states))
rk_iae = np.sum(np.abs(np_new_ref - np_rk_states))

print('PINN IAE:', pinn_iae)
print('Runge-Kutta IAE', rk_iae)

## Plot
import util.plot as pp

plotter = pp.Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_rk_controls[:, 0]],
             labels=['PINN', 'Runge-Kutta'],
             title='Van der Pol control signal',
             x_label='Time',
             y_label=None,
             draw_styles='steps',
             np_c_base=None)
# Obs: Here we have switched x1 and x2 to plot according to the text notation
plotter.plot(x_axis=np_t,
             y_axis_list=[np_states[:, 1], np_states[:, 0], np_rk_states[:, 1], np_rk_states[:, 0], np_new_ref[:, 1]],
             labels=['PINN X1', 'PINN X2', 'RK X1', 'RK X2', None],
             title='Van der Pol state vars',
             x_label='Time',
             y_label=None,
             line_styles=['-', '-', '-', '-', '--'],
             np_c_base=None)
plotter.show()
