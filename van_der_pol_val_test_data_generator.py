import numpy as np
import pandas as pd
from util.systems import VanDerPolSystem


# Saving validation and test data in a CSV file
# Parameters
random_seed = 10

sim_time = 10.0
u_change_t = 0.5
scenarios = 1
collocation_points_per_u = 10

# Constraints
lowest_u = -1.0
highest_u = 1.0
lowest_x = -3.0
highest_x = 3.0

# File name
if scenarios > 1:
    file_name = 'rand_seed_' + str(random_seed) + \
                '_sim_time_' + str(sim_time) + 's_' + \
                str(scenarios) + '_scenarios_' + \
                str(int((sim_time / u_change_t) * (collocation_points_per_u if collocation_points_per_u > 2 else 1))) + \
                '_collocation_points'
else:
    file_name = 'rand_seed_' + str(random_seed) + \
                '_sim_time_' + str(sim_time) + 's_' + \
                str(int((sim_time / u_change_t) * (collocation_points_per_u if collocation_points_per_u > 2 else 1))) + \
                '_collocation_points'

# Random seed
np.random.seed(random_seed)

# System simulator
simulator = VanDerPolSystem()

# Controls and initial conditions
np_us = np.random.uniform(low=lowest_u, high=highest_u, size=(scenarios, int(sim_time / u_change_t)))
np_x0s = np.random.uniform(low=lowest_x, high=highest_x, size=(scenarios, 2))

# Data generation
data = {'scenario': [],
        't': [],
        'u': [],
        'x1': [],
        'x2': []}

for j in range(scenarios):
    np_T = np.linspace(0, u_change_t, collocation_points_per_u)
    np_t = np_T

    np_u = np.tile(np_us[j, 0], (collocation_points_per_u, 1))

    np_x = simulator.run(np_T, np_us[j, 0], np_x0s[j, :])

    for i in range(1, int(sim_time / u_change_t)):
        np_t = np.append(np_t, np_T[1:] + np_t[-1])
        np_u = np.append(np_u[:-1],
                         np.tile(np_us[j, i], (collocation_points_per_u, np_u.shape[1])),
                         axis=0)
        np_x = np.append(np_x,
                         simulator.run(np_T, np_us[j, i], np_x[-1, :], output_t0=False),
                         axis=0)

    data['scenario'].append(np.tile(j + 1, (np_t.size,)))
    data['t'].append(np_t)
    data['u'].append(np_u[:, 0])
    data['x1'].append(np_x[:, 0])
    data['x2'].append(np_x[:, 1])

# Saving data
data['scenario'] = np.concatenate(data['scenario'])
data['t'] = np.concatenate(data['t'])
data['u'] = np.concatenate(data['u'])
data['x1'] = np.concatenate(data['x1'])
data['x2'] = np.concatenate(data['x2'])

df = pd.DataFrame(data)
df.to_csv('data/van_der_pol/' + file_name + '.csv', index=False)
