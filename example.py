import numpy as np
from four_tanks_system import ResponseAnalyser, CasadiSimulator

# Random seed
np.random.seed(30)

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

# Controls and initial conditions for training and testing
train_points = 1000
train_vs = 3.0 * np.random.rand(2, train_points)
train_ics = 20.0 * np.random.rand(4, train_points)

test_points = 100
test_vs = 3.0 * np.random.rand(2, test_points)
test_ics = 20.0 * np.random.rand(4, test_points)

# Neural network's working period
resp_an = ResponseAnalyser(sys_params)
ol_sample_time = resp_an.get_ol_sample_time(np.concatenate([train_vs, test_vs], axis=1))
np_t = np.linspace(0, ol_sample_time, 100)

# Training
pass

# Testing
simulator = CasadiSimulator(sys_params)
sampled_outputs = []
for i in range(test_vs.shape[1]):
    np_v = test_vs[:, i]
    np_ic = test_ics[:, i]
    np_h = simulator.run(np_t, np_v, np_ic)
    sampled_outputs.append(np_h)

# Plotting
# plt.plot(np_t, np_h[0], label='h1')
# plt.plot(np_t, np_h[1], label='h2')
# plt.plot(np_t, np_h[2], label='h3')
# plt.plot(np_t, np_h[3], label='h4')
# plt.legend()
# plt.show()
