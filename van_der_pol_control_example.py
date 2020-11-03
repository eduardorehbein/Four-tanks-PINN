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
use_runge_kutta = False

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Load model
model = VanDerPolPINN(hidden_layers=4, units_per_layer=20)

model.load('models/van_der_pol/2020-10-25-04-19-43-0dot5s-4l-20n-exhausted-model')

# Instance simulator
simulator = VanDerPolSystem()

# Reference adjustment
tile_points = int(sim_time / T / np_ref.shape[0])
new_np_ref = np.tile(np_ref[0, :], (tile_points, 1))
for i in range(1, np_ref.shape[0]):
    new_np_ref = np.append(new_np_ref, np.tile(np_ref[i, :], (tile_points, 1)), axis=0)

# Control
controller = PINNController(model, simulator)
np_t, np_controls, np_new_ref, np_states = controller.control(new_np_ref, np_x0, np_min_u, np_max_u, np_min_x, np_max_x,
                                                              sim_time, prediction_horizon, T, collocation_points_per_T,
                                                              use_runge_kutta=use_runge_kutta)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0]],
             labels=[None],
             title='Van der Pol control signal',
             x_label='Time',
             y_label=None,
             draw_styles='steps',
             np_c_base=None)
# Obs: Here we have switched x1 and x2 to plot according to the text notation
plotter.plot(x_axis=np_t,
             y_axis_list=[np_states[:, 1], np_states[:, 0], np_new_ref[:, 1]],
             labels=['$x_1$', '$x_2$', None],
             title='Van der Pol state vars',
             x_label='Time',
             y_label=None,
             line_styles=['-', '-', '--'],
             np_c_base=None)
plotter.show()
