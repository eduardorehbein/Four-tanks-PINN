import numpy as np
import pandas as pd
from util.systems.four_tanks_system import CasadiSimulator


# Parameters
random_seed = 30

scenarios = 1000
collocation_points = 10
T = 15.0

lowest_v = 0.5
highest_v = 3.0
lowest_h = 2.0
highest_h = 20.0

file_name = 'rand_seed_' + str(random_seed) + \
            '_T_' + str(T) + 's_' + \
            str(scenarios) + '_scenarios_' + \
            str(collocation_points) + '_collocation_points'

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
vs = np.random.uniform(low=lowest_v, high=highest_v, size=(scenarios, 2))
ics = np.random.uniform(low=lowest_h, high=highest_h, size=(scenarios, 4))

# Time
t = np.linspace(0, T, collocation_points)

# Data
h = simulator.run(t, vs[0], ics[0])
data = {'scenario': np.tile(1, (collocation_points,)),
        't': t,
        'v1': np.tile(vs[0, 0], (collocation_points,)),
        'v2': np.tile(vs[0, 1], (collocation_points,)),
        'h1_0': np.tile(ics[0, 0], (collocation_points,)),
        'h2_0': np.tile(ics[0, 1], (collocation_points,)),
        'h3_0': np.tile(ics[0, 2], (collocation_points,)),
        'h4_0': np.tile(ics[0, 3], (collocation_points,)),
        'h1': h[:, 0],
        'h2': h[:, 1],
        'h3': h[:, 2],
        'h4': h[:, 3]
        }

for i in range(1, scenarios):
    scenario = np.tile(i + 1, (collocation_points,))
    v = np.tile(vs[i], (collocation_points, 1))
    ic = np.tile(ics[i], (collocation_points, 1))
    h = simulator.run(t, vs[i], ics[i])

    data['scenario'] = np.append(data['scenario'], scenario)
    data['t'] = np.append(data['t'], t)
    data['v1'] = np.append(data['v1'], v[:, 0])
    data['v2'] = np.append(data['v2'], v[:, 1])
    data['h1_0'] = np.append(data['h1_0'], ic[:, 0])
    data['h2_0'] = np.append(data['h2_0'], ic[:, 1])
    data['h3_0'] = np.append(data['h3_0'], ic[:, 2])
    data['h4_0'] = np.append(data['h4_0'], ic[:, 3])
    data['h1'] = np.append(data['h1'], h[:, 0])
    data['h2'] = np.append(data['h2'], h[:, 1])
    data['h3'] = np.append(data['h3'], h[:, 2])
    data['h4'] = np.append(data['h4'], h[:, 3])

df = pd.DataFrame(data)
df.to_csv('data/four_tanks/' + file_name + '.csv', index=False)
