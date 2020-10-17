import numpy as np
import pandas as pd
from util.systems.four_tanks_system import CasadiSimulator


# Parameters
random_seed = 10

sim_time = 150.0
u_change_t = 2.0
scenarios = 1
collocation_points_per_v = 10

lowest_v = 0.5
highest_v = 3.0
lowest_h = 2.0
highest_h = 20.0

if scenarios > 1:
    file_name = 'long_signal_rand_seed_' + str(random_seed) + \
                '_sim_time_' + str(sim_time) + 's_' + \
                str(scenarios) + '_scenarios_' + \
                str(int((sim_time / u_change_t) * (collocation_points_per_v if collocation_points_per_v > 2 else 1))) + \
                '_collocation_points'
else:
    file_name = 'long_signal_rand_seed_' + str(random_seed) + \
                '_sim_time_' + str(sim_time) + 's_' + \
                str(int((sim_time / u_change_t) * (collocation_points_per_v if collocation_points_per_v > 2 else 1))) + \
                '_collocation_points'

# Set random seed
np.random.seed(random_seed)

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

# System simulator
simulator = CasadiSimulator(sys_params)

# Controls and initial conditions
np_v1s = np.random.uniform(low=lowest_v, high=highest_v, size=(scenarios, int(sim_time / u_change_t)))
np_v2s = np.random.uniform(low=lowest_v, high=highest_v, size=(scenarios, int(sim_time / u_change_t)))
np_h0s = np.random.uniform(low=lowest_h, high=highest_h, size=(scenarios, 4))

# Generate data
data = {'scenario': [],
        't': [],
        'v1': [],
        'v2': [],
        'h1': [],
        'h2': [],
        'h3': [],
        'h4': []}

for j in range(scenarios):
    np_T = np.linspace(0, u_change_t, collocation_points_per_v)
    np_t = np_T

    np_v1 = np.tile(np_v1s[j, 0], (collocation_points_per_v, 1))
    np_v2 = np.tile(np_v2s[j, 0], (collocation_points_per_v, 1))
    np_v = np.transpose(np.array([np_v1[-1], np_v2[-1]]))

    np_h = simulator.run(np_T, np_v, np_h0s[j, :])

    for i in range(1, int(sim_time / u_change_t)):
        np_t = np.append(np_t, np_T[1:] + np_t[-1])

        np_v1 = np.append(np_v1[:-1],
                          np.tile(np_v1s[j, i], (collocation_points_per_v, np_v1.shape[1])),
                          axis=0)
        np_v2 = np.append(np_v2[:-1],
                          np.tile(np_v2s[j, i], (collocation_points_per_v, np_v2.shape[1])),
                          axis=0)
        np_v = np.array([np_v1[-1], np_v2[-1]])
        np_h = np.append(np_h,
                         simulator.run(np_T, np_v, np_h[-1, :], output_t0=False),
                         axis=0)

    data['scenario'].append(np.tile(j + 1, (np_t.size,)))
    data['t'].append(np_t)
    data['v1'].append(np_v1[:, 0])
    data['v2'].append(np_v2[:, 0])
    data['h1'].append(np_h[:, 0])
    data['h2'].append(np_h[:, 1])
    data['h3'].append(np_h[:, 2])
    data['h4'].append(np_h[:, 3])

# Save data
data['scenario'] = np.concatenate(data['scenario'])
data['t'] = np.concatenate(data['t'])
data['v1'] = np.concatenate(data['v1'])
data['v2'] = np.concatenate(data['v2'])
data['h1'] = np.concatenate(data['h1'])
data['h2'] = np.concatenate(data['h2'])
data['h3'] = np.concatenate(data['h3'])
data['h4'] = np.concatenate(data['h4'])

df = pd.DataFrame(data)
df.to_csv('data/four_tanks/' + file_name + '.csv', index=False)
