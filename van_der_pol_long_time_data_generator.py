import numpy as np
import pandas as pd
from util.systems.van_der_pol_system import CasadiSimulator


# Parameters
random_seed = 30

collocation_points = 10000
t_range = 10.0
v_change_t = 2.0

lowest_u = -1.0
highest_u = 1.0
lowest_x = np.array([-1.5, -2.0])
highest_x = np.array([1.5, 2.0])

file_name = 'long_signal_rand_seed_' + str(random_seed) + \
            '_t_range_' + str(t_range) + 's_' + \
            str(collocation_points) + '_collocation_points'

# Set random seed
np.random.seed(random_seed)

# System simulator
simulator = CasadiSimulator()

# Controls and initial conditions for training and testing
us = np.random.uniform(low=lowest_u, high=highest_u, size=(int(t_range / v_change_t),))
ic = (highest_x - lowest_x)*np.random.rand(1, 2) + lowest_x

# Time
t = np.linspace(0, t_range, collocation_points)
t_change_index_step = int(v_change_t/(t_range/collocation_points))

# Data
x = simulator.run(t[:t_change_index_step], us[0], ic)
data = {'t': t,
        'u': np.tile(us[0], (t_change_index_step,)),
        'x1': x[:, 0],
        'x2': x[:, 1]}

for i in range(1, len(us)):
    u = np.tile(us[i], (t_change_index_step,))
    x1_0 = data['x1'].reshape((data['x1'].shape[0], 1))
    x2_0 = data['x2'].reshape((data['x2'].shape[0], 1))
    x = simulator.run(t[:t_change_index_step], us[i], np.concatenate([x1_0, x2_0], axis=1)[-1])

    data['u'] = np.append(data['u'], u)
    data['x1'] = np.append(data['x1'], x[:, 0])
    data['x2'] = np.append(data['x2'], x[:, 1])

df = pd.DataFrame(data)
df.to_csv('data/van_der_pol/' + file_name + '.csv', index=False)
