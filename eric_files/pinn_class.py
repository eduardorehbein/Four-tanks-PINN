''' author: Eric A. Antonelo '''

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Any, Union

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
import sys, os
from data_esp import prepare_data

from tensorflow.keras.layers import Dense
import factory_lbfgs


class PINN(object):

    def __init__(self, n_outputs, hidden_layers, units_per_layers, lb, ub, u_ub, u_lb, learning_rate=0.001):
        self.dtype = 'float64'
        tf.keras.backend.set_floatx(self.dtype)
        # Domain bounds (input: time t)
        self.lb = lb
        self.ub = ub
        # u bounds (output of network)
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.input_dimension = 1  # time t

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.input_dimension,)))
        self.model.add(tf.keras.layers.Lambda(
            lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))  # to [-1,1]
        for _ in range(hidden_layers):
            self.model.add(Dense(units_per_layers, 'tanh', kernel_initializer="glorot_normal"))
        self.model.add(Dense(n_outputs, None, kernel_initializer="glorot_normal"))

        # optimizer = tf.keras.optimizers.RMSprop(0.01)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, epsilon=None)  # (0.01)

    def predict(self, X):
        '''
        Predict the output u(t)
        :param X: data input points representing time t
        :return: u(t)
        '''
        X = self.tensor(X)
        #return self.model(X)
        return (1 + self.model(X)) * (self.u_ub - self.u_lb) / 2.0 + self.u_lb

    def set_opt_params(self, lr, beta_1=0.9, epsilon=None):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, epsilon=epsilon)

    def set_data(self, X_u, u, X_f):
        '''
        Prepare data: input points X_u, outputs of network u(t), collocation points X_f
        :param X_u: training data input points representing time t (initial condition of DAE)
        :param u:   training data output vectors u(t) of dimension TxM (T: total time; M: no. output variables)
        :param X_f:  collocation points for physics informed training (vector of times t)
        :return: None
        '''
        self.X_u = self.tensor(X_u)  # will be only time t for now
        self.u =  2 * (u - self.u_lb) / (self.u_ub - self.u_lb) - 1  # normalize to [-1,1]
        print(u.shape, self.u.shape, self.u)
        self.u = self.tensor(self.u)
        self.t_f = self.tensor(X_f)
        # self.x_f = self.tensor(X_f[:, 0:1]) # not PDE
        # self.t_f = self.tensor(X_f[:, 1:2])

    def u_fn(self, t): # training=True):
        return self.model(t, 1)

    def mse_u(self, predicted_y, target_y):
        return tf.reduce_mean(tf.square(predicted_y - target_y))

    def mse_f(self, f_physics):
        return tf.reduce_mean(tf.square(f_physics))

    def f(self):
        '''
        Compute function physics informed f(t) for minimization
        :return: f(t)
        '''
        pass

    def loss_function(self, predictions, return_all_losses=False):
        #c = 33775719714212.0
        loss_1 = self.mse_u(predictions, self.u)
        loss_2 = [self.mse_f( f ) for f in self.f() ]
        # / c
        loss = loss_1
        for l2 in loss_2:
            loss += l2
        if return_all_losses:
            return loss, loss_1, loss_2
        return loss

    def train_adam_step(self):  #(inputs, outputs, collocation_inputs, optimizer):
        with tf.GradientTape() as tape:
            predictions = self.model(self.X_u)  # training=True
            loss, loss_1, loss_2 = self.loss_function(predictions, True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # for unknown DAE parameters, compute gradients with respect to those variables in f()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, loss_1, loss_2

    def train_adam(self, N=100):
        # train with ADAM optmizer
        for epoch in range(N):
            current_loss, l1, l2 = self.train_adam_step()
            print('Epoch %2d:, loss=%2.5f, %2.5f, %2.5f %2.5f %2.5f' %
                  (epoch, current_loss, l1, l2[0],l2[1],l2[2] ))

    def get_loss(self):
        def loss():
            predictions = self.model(self.X_u)
            return self.loss_function(predictions, return_all_losses=False)
        return loss

    def train_lbfgs(self):
        # train with l-bfgs
        func = factory_lbfgs.function_factory(self.model, self.get_loss())
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
        print("train the model with L-BFGS solver")
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=200, tolerance=1e-8)
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        # test
        loss, l1, l2 = self.loss_function(self.model(self.X_u), return_all_losses=True)
        print('loss=%2.5f, %2.5f, %2.5f' %
              (loss, l1, l2))

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)



class PINN_4tanks(PINN):
    def f(self):
        '''
        Compute function physics informed f(t) for minimization
        :return: f(t)
        '''


#filepath = os.path.join("data", "XXXX.mat")

# t, Exact_u, X_star, u_star, X_u_train, u_train, \
#    X_f_train, ub, lb, u_ub, u_lb = prepare_data(filepath, N_f=25)

#pinn = PINN_4tanks(n_outputs=3, hidden_layers=2, units_per_layers=20,
#                lb=lb, ub=ub, u_lb=u_lb, u_ub=u_ub, learning_rate=0.003)
#pinn.set_data(X_u_train, u_train, X_f_train)
#pinn.train_adam(N=5000)
#pinn.train_lbfgs()

# model.save_weights('./checkpoints/pinn1') # Epoch 1999:, loss=0.11606, 0.08253, 0.03354
# model.load_weights('./checkpoints/pinn1')

###
# Getting the model predictions
# u_pred = pinn.predict(X_star)

