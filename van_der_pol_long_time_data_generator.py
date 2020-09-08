import numpy as np
import pandas as pd
from util.systems.van_der_pol_system import CasadiSimulator


# Parameters
random_seed = 10

sim_time = 10.0
u_change_t = 2.0
collocation_points_per_u = 2000

lowest_u = -1.0
highest_u = 1.0
lowest_x = np.array([-3.0, -3.0])
highest_x = np.array([3.0, 3.0])

file_name = 'long_signal_rand_seed_' + str(random_seed) + \
            '_sim_time_' + str(sim_time) + 's_' + \
            str(int((sim_time / u_change_t) * collocation_points_per_u)) + '_collocation_points'

# Set random seed
np.random.seed(random_seed)

# System simulator
simulator = CasadiSimulator()

# Controls and initial conditions for training and testing
np_us = np.random.uniform(low=lowest_u, high=highest_u, size=(int(sim_time / u_change_t),))
np_x0 = (highest_x - lowest_x) * np.random.rand(1, 2) + lowest_x

# Generate data
np_T = np.linspace(0, u_change_t, collocation_points_per_u)
np_t = np_T

np_u = np.tile(np_us[0], (collocation_points_per_u, 1))

np_x = np_x0
np_x = np.append(np_x,
                 simulator.run(np_T, np_us[0], np_x[0], output_t0=False),
                 axis=0)

for i in range(1, int(sim_time / u_change_t)):
    np_t = np.append(np_t, np_T[1:] + np_t[-1])

    np_ic = np.reshape(np_x[-1], np_x0.shape)

    np_u = np.append(np_u,
                     np.tile(np_us[i], (collocation_points_per_u - 1, np_u.shape[1])),
                     axis=0)
    np_x = np.append(np_x,
                     simulator.run(np_T, np_us[i], np_x[-1], output_t0=False),
                     axis=0)

# Save data
data = {'t': np_t,
        'u': np_u[:, 0],
        'x1': np_x[:, 0],
        'x2': np_x[:, 1]}

df = pd.DataFrame({'t': np_t,
                   'u': np_u[:, 0],
                   'x1': np_x[:, 0],
                   'x2': np_x[:, 1]})
df.to_csv('data/van_der_pol/' + file_name + '.csv', index=False)
