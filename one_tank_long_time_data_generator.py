import numpy as np
import pandas as pd
from util.systems.one_tank_system import CasadiSimulator


# Hyperparameters
random_seed = 30

collocation_points = 160000
t_range = 160.0
v_change_t = 20.0

lowest_v = 0.5
highest_v = 4.45
lowest_h = 2.0
highest_h = 20.0

file_name = 'long_signal_rand_seed_' + str(random_seed) + \
            '_t_range_' + str(t_range) + 's_' + \
            str(collocation_points) + '_collocation_points'

# Set random seed
np.random.seed(random_seed)

# System simulator
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14,  # [cm^3/Vs]
              }
simulator = CasadiSimulator(sys_params)

# Controls and initial conditions for training and testing
vs = np.random.uniform(low=lowest_v, high=highest_v, size=(int(t_range / v_change_t),))
ic = (highest_h - lowest_h)*np.random.rand() + lowest_h

# Time
t = np.linspace(0, t_range, collocation_points)
t_change_index_step = int(v_change_t/(t_range/collocation_points))

# Data
data = {'t': t,
        'v': np.tile(vs[0], (t_change_index_step,)),
        'h': simulator.run(t[:t_change_index_step], vs[0], ic)[0]}

for i in range(1, len(vs)):
    v = np.tile(vs[i], (t_change_index_step,))
    h = simulator.run(t[:t_change_index_step], vs[i], data['h'][-1])[0]

    data['v'] = np.append(data['v'], v)
    data['h'] = np.append(data['h'], h)

df = pd.DataFrame(data)
df.to_csv('data/one_tank/' + file_name + '.csv', index=False)
