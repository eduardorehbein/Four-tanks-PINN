import numpy as np
import pandas as pd
from util.systems.van_der_pol_system import CasadiSimulator


# Parameters
random_seed = 30

scenarios = 1000
collocation_points = 100
T = 8.0

lowest_u = -1.0
highest_u = 1.0
lowest_x = [-3.0, -3.0]
highest_x = [3.0, 3.0]

file_name = 'rand_seed_' + str(random_seed) + \
            '_T_' + str(T) + 's_' + \
            str(scenarios) + '_scenarios_' + \
            str(collocation_points) + '_collocation_points'

# Set random seed
np.random.seed(random_seed)

# System simulator
simulator = CasadiSimulator()

# Controls and initial conditions for training and testing
us = np.random.uniform(low=lowest_u, high=highest_u, size=(scenarios,))
ics = np.random.uniform(low=lowest_x, high=highest_x, size=(scenarios, 2))

# Neural network's max working period
t = np.linspace(0, T, collocation_points)

# Data
x = simulator.run(t, us[0], ics[0])
data = {'scenario': np.tile(1, (collocation_points,)),
        't': t,
        'u': np.tile(us[0], (collocation_points,)),
        'x1_0': np.tile(ics[0, :][0], (collocation_points,)),
        'x2_0': np.tile(ics[0, :][1], (collocation_points,)),
        'x1': x[:, 0],
        'x2': x[:, 1]}

for i in range(1, scenarios):
    scenario = np.tile(i + 1, (collocation_points,))
    u = np.tile(us[i], (collocation_points,))
    ic = np.tile(ics[i], (collocation_points, 2))
    x = simulator.run(t, us[i], ics[i])

    data['scenario'] = np.append(data['scenario'], scenario)
    data['t'] = np.append(data['t'], t)
    data['u'] = np.append(data['u'], u)
    data['x1_0'] = np.append(data['x1_0'], ic[:, 0])
    data['x2_0'] = np.append(data['x2_0'], ic[:, 1])
    data['x1'] = np.append(data['x1'], x[:, 0])
    data['x2'] = np.append(data['x2'], x[:, 1])

df = pd.DataFrame(data)
df.to_csv('data/van_der_pol/' + file_name + '.csv', index=False)
