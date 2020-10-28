import numpy as np
import tensorflow as tf
from util.plot import Plotter
from util.controller import PINNController
from util.pinn import OneTankPINN
from util.systems import OneTankSystem


# Parameters
np_h0 = np.array([[5.0]])
np_ref = np.array([[10.0]])

np_min_u = np.array([[0.0]])
np_max_u = np.array([[4.45]])

T = 10.0
collocation_points_per_T = 100
prediction_horizon = 5*T
sim_time = 50.0

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Instance PINN
model = OneTankPINN(sys_params=sys_params, hidden_layers=2, units_per_layer=10)
model.load('models/one_tank/2020-08-06-11-24-21-10s-2l-10n-best-model')

# Instance simulator
simulator = OneTankSystem(sys_params)

# Control
controller = PINNController(model, simulator)
np_t, np_controls, np_states = controller.control(np_ref, np_h0, np_min_u, np_max_u,
                                                  sim_time, prediction_horizon, T, collocation_points_per_T)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_states[:, 0], np.tile(np_ref[0, 0], (np_states.shape[0],))],
             labels=['$u$', '$h$', None],
             title='Van der Pol control via PINN',
             x_label='Time',
             y_label=None,
             line_styles=['-', '-', '--'],
             draw_styles=['steps', 'default', 'default'])
plotter.show()
