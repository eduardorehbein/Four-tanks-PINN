import numpy as np
import pandas as pd
from util.systems.one_tank_system import CasadiSimulator


# Hyperparameters
random_seed = 30

scenarios = 1100
collocation_points = 100
t_range = 10.0

lowest_v = 0.5
highest_v = 4.45
lowest_h = 2.0
highest_h = 20.0

file_name = 'rand_seed_' + str(random_seed) + \
            '_t_range_' + str(t_range) + 's_' + \
            str(scenarios) + '_scenarios_' + \
            str(collocation_points) + '_collocation_points'

# Setting random seed
np.random.seed(random_seed)

# System simulator
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14,  # [cm^3/Vs]
              }
simulator = CasadiSimulator(sys_params)

# Controls and initial conditions for training and testing
vs = np.random.uniform(low=lowest_v, high=highest_v, size=(scenarios,))
ics = np.random.uniform(low=lowest_h, high=highest_h, size=(scenarios,))

# Neural network's working period
t = np.linspace(0, t_range, collocation_points)

# Data
data = {'scenario': np.tile(1, (collocation_points,)),
        't': t,
        'v': np.tile(vs[0], (collocation_points,)),
        'ic': np.tile(ics[0], (collocation_points,)),
        'h': simulator.run(t, vs[0], ics[0])}

for i in range(1, scenarios):
    scenario = np.tile(i + 1, (collocation_points,))
    v = np.tile(vs[i], (collocation_points,))
    ic = np.tile(ics[i], (collocation_points,))
    h = simulator.run(t, vs[i], ics[i])

    data['scenario'] = np.append(data['scenario'], scenario)
    data['t'] = np.append(data['t'], t)
    data['v'] = np.append(data['v'], v)
    data['ic'] = np.append(data['ic'], ic)
    data['h'] = np.append(data['h'], h)

train_df = pd.DataFrame(data)
train_df.to_csv('data/one_tank/' + file_name + '.csv', index=False)
