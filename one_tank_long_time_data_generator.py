import numpy as np
import pandas as pd
from util.systems import OneTankSystem


# Parameters
random_seed = 30

sim_time = 160.0
v_change_t = 20.0
collocation_points_per_v = 200

lowest_v = 0.5
highest_v = 4.45
lowest_h = 2.0
highest_h = 20.0

file_name = 'long_signal_rand_seed_' + str(random_seed) + \
            '_sim_time_' + str(sim_time) + 's_' + \
            str(int((sim_time / v_change_t) * collocation_points_per_v)) + '_collocation_points'

# Set random seed
np.random.seed(random_seed)

# System simulator
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14,  # [cm^3/Vs]
              }
simulator = OneTankSystem(sys_params)

# Controls and initial conditions
np_vs = np.random.uniform(low=lowest_v, high=highest_v, size=(int(sim_time / v_change_t),))
np_h0 = (highest_h - lowest_h) * np.random.rand(1, 1) + lowest_h

# Generate data
np_T = np.linspace(0, v_change_t, collocation_points_per_v)
np_t = np_T

np_v = np.tile(np_vs[0], (collocation_points_per_v, 1))

np_h = np_h0
np_h = np.append(np_h,
                 simulator.run(np_T, np_vs[0], np_h[0], output_t0=False),
                 axis=0)

for i in range(1, int(sim_time / v_change_t)):
    np_t = np.append(np_t, np_T[1:] + np_t[-1])
    np_v = np.append(np_v,
                     np.tile(np_vs[i], (collocation_points_per_v - 1, np_v.shape[1])),
                     axis=0)
    np_h = np.append(np_h,
                     simulator.run(np_T, np_vs[i], np_h[-1], output_t0=False),
                     axis=0)

# Save data
df = pd.DataFrame({'t': np_t,
                   'v': np_v[:, 0],
                   'h': np_h[:, 0]})
df.to_csv('data/one_tank/' + file_name + '.csv', index=False)
