import numpy as np
from util.controller import PINNController
from util.pinn import FourTanksPINN
from util.systems import FourTanksSystem
from util.plot import Plotter


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

# System parameters dictionary
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

# PINN Instance
model = FourTanksPINN(sys_params=sys_params,
                      hidden_layers=8,
                      units_per_layer=20)

# Loading model
model.load('models/four_tanks/2020-11-06-04-46-56-10dot0s-8l-20n-exhausted-model')
model.trained_T = 10.0

# System simulator
simulator = FourTanksSystem(sys_params)

# Reference adjustment
tile_points = int(sim_time / T / np_ref.shape[0])
np_adj_ref = np.tile(np_ref[0, :], (tile_points, 1))
for i in range(1, np_ref.shape[0]):
    np_adj_ref = np.append(np_adj_ref, np.tile(np_ref[i, :], (tile_points, 1)), axis=0)

# Control
controller = PINNController(model, simulator)

# Control example using PINN
np_t, np_controls, np_new_ref, np_states = controller.control(np_adj_ref, np_h0, np_min_v, np_max_v, np_min_h, np_max_h,
                                                              sim_time, prediction_horizon, T, collocation_points_per_T,
                                                              outputs_to_control)

# Control example using Runge-Kutta
np_rk_t, np_rk_controls, np_rk_new_ref, np_rk_states = controller.control(np_adj_ref, np_h0,
                                                                          np_min_v, np_max_v, np_min_h, np_max_h,
                                                                          sim_time, prediction_horizon, T,
                                                                          collocation_points_per_T,
                                                                          outputs_to_control, True)

# IAEs
pinn_iae = np.sum(np.abs(np_new_ref[:, :2] - np_states[:, :2]))
rk_iae = np.sum(np.abs(np_new_ref[:, :2] - np_rk_states[:, :2]))

print('PINN IAE:', pinn_iae)
print('Runge-Kutta IAE', rk_iae)

# Plot
plotter = Plotter()
plotter.plot(x_axis=np_t,
             y_axis_list=[np_controls[:, 0], np_controls[:, 1], np_rk_controls[:, 0], np_rk_controls[:, 1]],
             labels=['PINN V1', 'PINN V2', 'RK V1', 'RK V2'],
             title='Four tanks\' control signals',
             x_label='Time',
             y_label=None,
             draw_styles='steps',
             np_c_base=None)
plotter.multiplot(x_axis=np_t,
                  y_axis_matrices=[[np_states[:, 2], np_states[:, 3], np_rk_states[:, 2], np_rk_states[:, 3]],
                                   [np_states[:, 0], np_states[:, 1], np_rk_states[:, 0], np_rk_states[:, 1],
                                    np_new_ref[:, 0], np_new_ref[:, 1]]],
                  labels_list=[['PINN H3', 'PINN H4', 'RK H3', 'RK H4'],
                               ['PINN H1', 'PINN H2', 'RK H1', 'RK H2', None, None]],
                  title='Four tanks\' levels',
                  x_label='Time',
                  y_labels_list=[None, None],
                  line_styles=[['-', '-', '-', '-'],
                               ['-', '-', '-', '-', '--', '--']],
                  np_c_base=None)
plotter.show()
