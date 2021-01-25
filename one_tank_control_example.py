import numpy as np
import tensorflow as tf
from util.plot import Plotter
from util.controller import PINNController
from util.pinn import OneTankPINN
from util.systems import OneTankSystem


# Constraints
np_min_v = np.array([[0.0]])
np_max_v = np.array([[4.45]])

np_min_h = np.array([[2.0]])
np_max_h = np.array([[20.0]])

# Initial condition
np_h0 = np.array([[5.0]])

# Reference
np_ref = np.array([[10.0],
                   [18.0],
                   [5.0],
                   [8.0],
                   [12.0]])

# Controller and simulation parameters
T = 10.0
collocation_points_per_T = 10
prediction_horizon = 5*T
sim_time = 1000.0

# Parallel threads setup
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# System parameters dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# PINN Instance
model = OneTankPINN(sys_params=sys_params, hidden_layers=2, units_per_layer=10)
model.load('models/one_tank/2020-08-06-11-24-21-10s-2l-10n-best-model')

# Simulator
simulator = OneTankSystem(sys_params)

# Reference adjustment
tile_points = int(sim_time / T / np_ref.shape[0])
np_adj_ref = np.tile(np_ref[0, :], (tile_points, 1))
for i in range(1, np_ref.shape[0]):
    np_adj_ref = np.append(np_adj_ref, np.tile(np_ref[i, :], (tile_points, 1)), axis=0)

# Controller
controller = PINNController(model, simulator)

# PINN control example
np_t, np_controls, np_new_ref, np_states = controller.control(np_adj_ref, np_h0, np_min_v, np_max_v, np_min_h, np_max_h,
                                                              sim_time, prediction_horizon, T, collocation_points_per_T)

# Runge-Kutta control example
np_rk_t, np_rk_controls, np_rk_new_ref, np_rk_states = controller.control(np_adj_ref, np_h0,
                                                                          np_min_v, np_max_v, np_min_h, np_max_h,
                                                                          sim_time, prediction_horizon, T,
                                                                          collocation_points_per_T,
                                                                          use_runge_kutta=True)

# IAEs
pinn_iae = np.sum(np.abs(np_new_ref - np_states))
rk_iae = np.sum(np.abs(np_new_ref - np_rk_states))

print('PINN IAE:', pinn_iae)
print('Runge-Kutta IAE', rk_iae)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_rk_controls[:, 0], np_states[:, 0], np_rk_states[:, 0], np_new_ref[:, 0]],
             labels=['PINN V', 'RK V', 'PINN H', 'RK H', None],
             title='One tank control',
             x_label='Time',
             y_label=None,
             line_styles=['-', '-', '-', '-', '--'],
             draw_styles=['steps', 'steps', 'default', 'default', 'default'],
             np_c_base=None)
plotter.show()
