import numpy as np
import tensorflow as tf
import pandas as pd
from util.tests import WorkingPeriodTester
from util.pinn import OneTankPINN

# Configure parallel threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Random seed
np.random.seed(30)

# System parameters' dictionary
sys_params = {'g': 981.0,  # [cm/s^2]
              'a': 0.071,  # [cm^2]
              'A': 28.0,  # [cm^2]
              'k': 3.14  # [cm^3/Vs]
              }

# Load data
working_periods_to_test = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
train_dfs = [pd.read_csv('data/one_tank/rand_seed_30_t_range_' + str(working_period) +
                         's_1105_scenarios_100_collocation_points.csv') for working_period in working_periods_to_test]
test_df = pd.read_csv('data/one_tank/long_signal_rand_seed_30_t_range_160.0s_160000_collocation_points.csv')

# Train parameters
adam_epochs = 5
max_lbfgs_iterations = 10

# Test
tester = WorkingPeriodTester(df_time_var='t',
                             df_scenario_var='scenario',
                             df_X_var=['t', 'v', 'ic'],
                             df_Y_var=['h'],
                             df_ic_var=['ic'],
                             adam_epochs=adam_epochs,
                             max_lbfgs_iterations=max_lbfgs_iterations)
tester.test(PINNModelClass=OneTankPINN,
            sys_params=sys_params,
            hidden_layers=2,
            units_per_layer=10,
            train_dfs=train_dfs,
            test_df=test_df,
            train_scenarios=1000,
            val_scenarios=100,
            results_subdirectory='one_tank')
