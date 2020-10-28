import numpy as np
import tensorflow as tf
from util.controller import PINNController
from util.pinn import FourTanksPINN
from util.systems import FourTanksSystem
from util.plot import Plotter


# Parameters
np_h0 = np.array([[1.0, 1.0, 1.0, 1.0]])
np_ref = np.array([[6.37, 9.88, 2.14, 1.74]])

np_min_v = np.array([[0.5, 0.5]])
np_max_v = np.array([[3.0, 3.0]])

T = 10.0
collocation_points_per_T = 10
prediction_horizon = 15*T
sim_time = 500.0

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

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
# Instance PINN
model = FourTanksPINN(sys_params=sys_params,
                      hidden_layers=5,
                      units_per_layer=20)

# Load model
model.load('models/four_tanks/2020-10-27-18-39-55-10dot0s-5l-20n-exhausted-model')
model.trained_T = 10.0

# Instance simulator
simulator = FourTanksSystem(sys_params)

# Control
controller = PINNController(model, simulator)
np_t, np_controls, np_states = controller.control(np_ref, np_h0, np_min_v, np_max_v,
                                                  sim_time, prediction_horizon, T, collocation_points_per_T)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_controls[:, 1]],
             labels=['$v_1$', '$v_2$'],
             title='Four tanks\' control signals',
             x_label='Time',
             y_label='Inputs',
             draw_styles='steps')
plotter.plot(x_axis=np_t,
             y_axis_list=[np_states[:, 0],
                          np_states[:, 1],
                          np_states[:, 2],
                          np_states[:, 3],
                          np.tile(np_ref[0, 0], (np_states.shape[0],)),
                          np.tile(np_ref[0, 1], (np_states.shape[0],)),
                          np.tile(np_ref[0, 2], (np_states.shape[0],)),
                          np.tile(np_ref[0, 3], (np_states.shape[0],))],
             labels=['$h_1$', '$h_2$', '$h_3$', '$h_4$', None, None, None, None],
             title='Four tanks\' levels',
             x_label='Time',
             y_label='Levels',
             line_styles=['-', '-', '-', '-', '--', '--', '--', '--'])
plotter.show()
