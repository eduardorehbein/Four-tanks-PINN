import numpy as np
import matplotlib.pyplot as plt
from four_tanks_system import ResponseAnalyser, CasadiSimulator
from normalizer import Normalizer
from pinn import FourTanksPINN

# Random seed
np.random.seed(30)

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

# Controls and initial conditions for training and testing
train_points = 1000
np_train_vs = 3.0 * np.random.rand(2, train_points)
np_train_ics = 20.0 * np.random.rand(4, train_points)

test_points = 100
np_test_vs = 3.0 * np.random.rand(2, test_points)
np_test_ics = 20.0 * np.random.rand(4, test_points)

# Neural network's working period
resp_an = ResponseAnalyser(sys_params)
ol_sample_time = resp_an.get_ol_sample_time(np.concatenate([np_train_vs, np_test_vs], axis=1))
# ol_sample_time = 50
np_t = np.array([np.linspace(0, ol_sample_time, 100)])

# Training data
np_train_u_t = np.zeros((1, np_train_ics.shape[1]))
np_train_u_v = np_train_vs
np_train_u_ic = np_train_ics

np_train_f_t = None
np_train_f_v = None
np_train_f_ic = None

for j in range(np_train_vs.shape[1]):
    np_ic = np.transpose(np.tile(np_train_ics[:, j], (np_t.shape[1], 1)))
    np_v = np.transpose(np.tile(np_train_vs[:, j], (np_t.shape[1], 1)))

    if np_train_f_t is None:
        np_train_f_t = np_t
    else:
        np_train_f_t = np.append(np_train_f_t, np_t, axis=1)
    if np_train_f_v is None:
        np_train_f_v = np_v
    else:
        np_train_f_v = np.append(np_train_f_v, np_v, axis=1)
    if np_train_f_ic is None:
        np_train_f_ic = np_ic
    else:
        np_train_f_ic = np.append(np_train_f_ic, np_ic, axis=1)

# Normalizers
t_normalizer = Normalizer()
v_normalizer = Normalizer()
h_normalizer = Normalizer()

t_normalizer.parametrize(np_t)
v_normalizer.parametrize(np_train_vs)
h_normalizer.parametrize(np_train_ics)

# Train data normalization
np_norm_train_u_t = t_normalizer.normalize(np_train_u_t)
np_norm_train_u_v = v_normalizer.normalize(np_train_u_v)
np_norm_train_u_ic = h_normalizer.normalize(np_train_u_ic)

np_norm_train_f_t = t_normalizer.normalize(np_train_f_t)
np_norm_train_f_v = v_normalizer.normalize(np_train_f_v)
np_norm_train_f_ic = h_normalizer.normalize(np_train_f_ic)

# PINN instancing
hidden_layers = [10, 10, 10, 10]
learning_rate = 0.001
model = FourTanksPINN(sys_params=sys_params,
                      hidden_layers=hidden_layers,
                      learning_rate=learning_rate,
                      t_normalizer=t_normalizer,
                      v_normalizer=v_normalizer,
                      h_normalizer=h_normalizer)

# Training
max_epochs = 20000
stop_loss = 0.001
# model.train(np_norm_train_u_t, np_norm_train_u_v, np_norm_train_u_ic, np_norm_train_f_t, np_norm_train_f_v,
#             np_norm_train_f_ic, max_epochs=max_epochs, stop_loss=stop_loss)

# Testing
simulator = CasadiSimulator(sys_params)
sampled_outputs = []
predictions = []
np_norm_t = t_normalizer.normalize(np_t)
for i in range(np_test_vs.shape[1]):
    np_v = np_test_vs[:, i]
    np_ic = np_test_ics[:, i]

    np_h = simulator.run(np_t, np_v, np_ic)
    sampled_outputs.append(np_h)

np_h = sampled_outputs[0]

# Plotting
plt.figure(1)
plt.title('Levels h1(t) vs nn1(t)')
plt.plot(np_t[0], np_h[0], label='h1')
plt.plot(np_t[0], np_h[0], label='nn1')
plt.legend()
plt.figure(2)
plt.title('Levels h2(t) vs nn2(t)')
plt.plot(np_t[0], np_h[1], label='h2')
plt.plot(np_t[0], np_h[1], label='nn2')
plt.legend()
plt.figure(3)
plt.title('Levels h3(t) vs nn3(t)')
plt.plot(np_t[0], np_h[2], label='h3')
plt.plot(np_t[0], np_h[2], label='nn3')
plt.legend()
plt.figure(4)
plt.title('Levels h4(t) vs nn4(t)')
plt.plot(np_t[0], np_h[3], label='h4')
plt.plot(np_t[0], np_h[3], label='nn4')
plt.legend()
plt.savefig('./results/plot.pdf')
