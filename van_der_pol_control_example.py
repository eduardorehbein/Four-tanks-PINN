import numpy as np
import tensorflow as tf
from util.controller import PINNController
from util.pinn import VanDerPolPINN
from util.systems import VanDerPolSystem
from util.plot import Plotter


# Parameters
np_x0 = np.array([[0.0, 1.0]])
np_ref = np.array([[0.0, 0.0]])

np_min_u = np.array([[-1.0]])
np_max_u = np.array([[1.0]])

T = 0.5
collocation_points_per_T = 10
prediction_horizon = 5*T
sim_time = 10.0

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Load model
model = VanDerPolPINN(hidden_layers=4, units_per_layer=20)

model.load('models/van_der_pol/2020-10-25-04-19-43-0dot5s-4l-20n-exhausted-model')

# Instance simulator
simulator = VanDerPolSystem()

# Control
controller = PINNController(model, simulator)
np_t, np_controls, np_states = controller.control(np_ref, np_x0, np_min_u, np_max_u,
                                                  sim_time, prediction_horizon, T, collocation_points_per_T)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_states[:, 0], np_states[:, 1],
                          np.tile(np_ref[0, 0], (np_states.shape[0],))],
             labels=['$u$', '$x_1$', '$x_2$', None],
             title='Van der Pol control via PINN',
             x_label='Time',
             y_label=None,
             line_styles=['-', '-', '-', '--'],
             draw_styles=['steps', 'default', 'default', 'default'])
plotter.show()
