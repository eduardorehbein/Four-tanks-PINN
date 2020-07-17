import copy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from util.factory_lbfgs import function_factory

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py


class PINN:
    def __init__(self, n_inputs, n_outputs, hidden_layers, units_per_layer, X_normalizer, Y_normalizer,
                 learning_rate=0.001):
        # Input and output vectors' dimension
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Normalizers
        self.X_normalizer = X_normalizer
        self.Y_normalizer = Y_normalizer

        # Model
        tf.keras.backend.set_floatx('float64')
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.n_inputs,)))
        self.model.add(tf.keras.layers.Lambda(lambda tf_X: self.X_normalizer.normalize(tf_X)))  # Normalize data
        for i in range(hidden_layers):
            self.model.add(tf.keras.layers.Dense(units_per_layer, 'tanh', kernel_initializer="glorot_normal"))
        self.model.add(tf.keras.layers.Dense(n_outputs, None, kernel_initializer="glorot_normal"))
        self.model.add(tf.keras.layers.Lambda(lambda tf_NN: self.Y_normalizer.denormalize(tf_NN)))  # Denormalize data

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = None
        self.set_opt_params(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        # Training losses
        self.train_f_loss = []
        self.train_u_loss = []
        self.train_total_loss = []

        # Validation loss
        self.validation_loss = []

    def predict(self, np_X, np_ic=None, working_period=None, time_column=0):
        '''
        Predict the output NN(X)
        :param np_X: numpy data input points X
        :return: NN(X)
        '''

        if np_ic is None:
            tf_X = self.tensor(np_X)
            tf_NN = self.model(tf_X)

            np_NN = tf_NN.numpy()
            if np_NN.shape[1] == 1:
                return np.transpose(np_NN)[0]
            else:
                return np_NN
        elif np_X.shape[1] + np_ic.shape[0] == self.n_inputs:
            if working_period is not None:
                np_Z = self.process_input(np_X, np_ic, working_period, time_column)
                tf_X = self.tensor(np_Z)
                tf_NN = self.model(tf_X)

                np_NN = tf_NN.numpy()
                if np_NN.shape[1] == 1:
                    return np.transpose(np_NN)[0]
                else:
                    return np_NN
            else:
                raise Exception('Missing neural network\'s working period.')
        else:
            raise Exception('np_X dimension plus np_ic dimension do not match neural network\'s input dimension')

    def process_input(self, np_X, np_ic, working_period, time_column):
        # TODO: Make it works with time steps bigger than working period
        np_Z = copy.deepcopy(np_X)
        np_Z[:, time_column] = np_Z[:, time_column] % working_period

        previous_t = np_Z[0, time_column]
        np_y0 = copy.deepcopy(np_ic)
        new_columns = []
        for index, row in enumerate(np_Z):
            if row[time_column] < previous_t:
                np_w = copy.deepcopy(np_Z[index-1, :])
                np_w[time_column] = working_period
                prediction_input = np.array([np.append(np_w, np_y0)])
                np_y0 = self.predict(prediction_input)
            previous_t = row[time_column]
            new_columns.append(np_y0)
        np_new_columns = np.array(new_columns)

        return np.append(np_Z, np_new_columns, axis=1)

    def tensor(self, np_X):
        return tf.convert_to_tensor(np_X, dtype=tf.dtypes.float64)

    def set_opt_params(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1,
                                                  beta_2=beta_2, epsilon=epsilon)

    def train(self, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
              adam_epochs=500, max_lbfgs_iterations=1000, epochs_per_print=100,
              u_loss_weight=1.0, f_loss_weight=0.1, save_losses=True):
        # Train data
        tf_train_u_X = self.tensor(np_train_u_X)
        tf_train_u_Y = self.tensor(np_train_u_Y)
        tf_train_f_X = self.tensor(np_train_f_X)

        # Validation data
        tf_val_X = self.tensor(np_val_X)
        tf_val_Y = self.tensor(np_val_Y)

        # Train with Adam
        self.train_adam(tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                        adam_epochs, epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

        # Train with L-BFGS
        self.train_lbfgs(tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                         max_lbfgs_iterations, epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

    def train_adam(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                   epochs, epochs_per_print, u_loss_weight, f_loss_weight, save_losses):
        # Train states and variables
        epoch = 0
        grads = None

        tf_val_loss = self.tensor(np.Inf)
        tf_best_val_loss = copy.deepcopy(tf_val_loss)

        best_weights = copy.deepcopy(self.model.get_weights())

        # Train process
        while epoch <= epochs:
            # Update weights and biases
            if grads is not None:
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Learning rate adjustments
            if epoch % 10000 == 0:
                self.learning_rate = self.learning_rate / 2
                self.set_opt_params(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

            # Network's gradients
            with tf.GradientTape(persistent=True) as tape:
                tf_total_loss, tf_u_loss, tf_f_loss = self.get_losses(tf_train_u_X, tf_train_u_Y, tf_train_f_X,
                                                                      u_loss_weight, f_loss_weight)
            grads = tape.gradient(tf_total_loss, self.model.trainable_variables)

            # Validation
            tf_val_NN = self.model(tf_val_X)
            tf_val_loss = tf.reduce_mean(tf.square(tf_val_NN - tf_val_Y))
            np_val_loss = tf_val_loss.numpy()

            if tf_val_loss < tf_best_val_loss:
                tf_best_val_loss = copy.deepcopy(tf_val_loss)
                best_weights = copy.deepcopy(self.model.get_weights())

            # Save loss values
            if save_losses:
                self.train_total_loss.append(tf_total_loss.numpy())
                self.train_u_loss.append(tf_u_loss.numpy())
                self.train_f_loss.append(tf_f_loss.numpy())

                self.validation_loss.append(np_val_loss)

            if epoch % epochs_per_print == 0:
                print('Epoch:', str(epoch), '-', 'Adam\'s validation loss:', str(np_val_loss))

            # Epoch count
            epoch = epoch + 1

        # Epoch adjustment
        epoch = epoch - 1

        # Set best weights
        self.model.set_weights(best_weights)

        # Print final validation loss
        print('Validation loss at the Adam\'s end -> Epoch:', str(epoch), '-',
              'validation loss:', tf_best_val_loss.numpy())

    def get_losses(self, tf_u_X, tf_u_Y, tf_f_X, u_loss_weight, f_loss_weight):
        tf_u_NN = self.model(tf_u_X)
        tf_u_loss = tf.reduce_mean(tf.square(tf_u_NN - tf_u_Y))

        tf_f_NN = self.f(tf_f_X)
        tf_f_loss = tf.reduce_mean(tf.square(tf_f_NN))

        tf_weighted_u_loss = u_loss_weight * tf_u_loss
        tf_weighted_f_loss = f_loss_weight * tf_f_loss
        tf_total_loss = tf_weighted_u_loss + tf_weighted_f_loss

        return tf_total_loss, tf_weighted_u_loss, tf_weighted_f_loss

    def f(self, tf_X):
        '''
        Compute function physics informed f(X) for minimization
        :return: f(X)
        '''

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as f_tape:
            f_tape.watch(tf_X)
            tf_NN = self.model(tf_X)
            np_output_selector = np.eye(self.n_outputs)
            decomposed_NN = []
            for i in range(self.n_outputs):
                tf_NN_single_output = tf.matmul(tf_NN, tf.transpose(self.tensor([np_output_selector[i]])))
                decomposed_NN.append(tf_NN_single_output)

        return self.expression(tf_X, tf_NN, decomposed_NN, f_tape)

    def train_lbfgs(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                    max_iterations, epochs_per_print, u_loss_weight, f_loss_weight, save_losses):
        func = function_factory(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                                epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

        # Convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
        res = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params,
                                           max_iterations=max_iterations)  # Each iteration is equivalent to 2-4 epochs

        # After training, the final optimized parameters are still in res.position
        # So we have to manually put them back to the model
        func.assign_new_model_parameters(res.position)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def get_weights(self):
        weights = []
        for layer in self.model.layers:
            weights.append(layer.get_weights())

        return weights

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        return self.tensor(0.0)


class OneTankPINN(PINN):
    def __init__(self, sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate=0.001):
        super().__init__(3, 1, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate)

        self.k = sys_params['k']
        self.a = sys_params['a']
        self.A = sys_params['A']
        self.two_g_sqrt = tf.sqrt(self.tensor(2 * sys_params['g']))

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        # ODE sys: dh_dt = (k/A)*v - (a/A)*sqrt(2*g*h)

        tf_v = tf.slice(tf_X, [0, 1], [tf_X.shape[0], 1])

        tf_dnn_dx = tape.gradient(tf_NN, tf_X)
        tf_dnn_dt = tf.slice(tf_dnn_dx, [0, 0], [tf_dnn_dx.shape[0], 1])

        return self.A * tf_dnn_dt + (self.a * self.two_g_sqrt) * tf.sqrt(tf.maximum(tf_NN, 0.0)) - self.k * tf_v


class FourTanksPINN(PINN):
    def __init__(self, sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate=0.001):
        super().__init__(7, 4, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate)

        # System parameters to matrix form
        self.B = []
        self.B.append(
            self.tensor([[sys_params['A1'], 0, 0, 0],
                         [0, sys_params['A2'], 0, 0],
                         [0, 0, sys_params['A3'], 0],
                         [0, 0, 0, sys_params['A4']]])
        )  # B[0]
        self.B.append(
            self.tensor([[sys_params['a1'], 0, - sys_params['a3'], 0],
                         [0, sys_params['a2'], 0, - sys_params['a4']],
                         [0, 0, sys_params['a3'], 0],
                         [0, 0, 0, sys_params['a4']]])
        )  # B[1]
        self.B.append(
            self.tensor([[sys_params['alpha1'] * sys_params['k1'], 0],
                         [0, sys_params['alpha2'] * sys_params['k2']],
                         [0, (1 - sys_params['alpha2']) * sys_params['k2']],
                         [(1 - sys_params['alpha1']) * sys_params['k1'], 0]])
        )  # B[2]

        self.two_g_sqrt = tf.sqrt(self.tensor(2 * sys_params['g']))

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        # ODE sys: dh1_dt = -(a1/A1)*sqrt(2*g*h1) + (a3/A1)*sqrt(2*g*h3) + ((alpha1*k1)/A1)*v1
        #          dh2_dt = -(a2/A2)*sqrt(2*g*h2) + (a4/A2)*sqrt(2*g*h4) + ((alpha2*k2)/A2)*v2
        #          dh3_dt = -(a3/A3)*sqrt(2*g*h3) + (((1 - alpha2)*k2)/A3)*v2
        #          dh4_dt = -(a4/A4)*sqrt(2*g*h4) + (((1 - alpha1)*k1)/A4)*v1
        #
        # In matrix form: B[0]*dot_H + sqrt(2*g)*B[1]*sqrt(H) - B[2]*V

        tf_v = tf.transpose(tf.slice(tf_X, [0, 1], [tf_X.shape[0], 2]))
        tf_nn = tf.transpose(tf_NN)

        tf_dnn1_dx = tape.gradient(decomposed_NN[0], tf_X)
        tf_dnn2_dx = tape.gradient(decomposed_NN[1], tf_X)
        tf_dnn3_dx = tape.gradient(decomposed_NN[2], tf_X)
        tf_dnn4_dx = tape.gradient(decomposed_NN[3], tf_X)

        tf_dnn_dt = tf.transpose(tf.concat([tf.slice(tf_dnn1_dx, [0, 0], [tf_dnn1_dx.shape[0], 1]),
                                            tf.slice(tf_dnn2_dx, [0, 0], [tf_dnn2_dx.shape[0], 1]),
                                            tf.slice(tf_dnn3_dx, [0, 0], [tf_dnn3_dx.shape[0], 1]),
                                            tf.slice(tf_dnn4_dx, [0, 0], [tf_dnn4_dx.shape[0], 1])], axis=1))

        tf_f_loss = tf.matmul(self.B[0], tf_dnn_dt) + self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(tf_nn)) - \
                    tf.matmul(self.B[2], tf_v)

        return tf.transpose(tf_f_loss)


class OldFourTanksPINN:
    def __init__(self, sys_params, hidden_layers, learning_rate,
                 t_normalizer=None, v_normalizer=None, h_normalizer=None):
        # System parameters to matrix form
        self.B = []
        self.B.append(
            tf.constant([[sys_params['A1'], 0, 0, 0],
                         [0, sys_params['A2'], 0, 0],
                         [0, 0, sys_params['A3'], 0],
                         [0, 0, 0, sys_params['A4']]], dtype=tf.float32)
        )  # B[0]
        self.B.append(
            tf.constant([[sys_params['a1'], 0, - sys_params['a3'], 0],
                         [0, sys_params['a2'], 0, - sys_params['a4']],
                         [0, 0, sys_params['a3'], 0],
                         [0, 0, 0, sys_params['a4']]], dtype=tf.float32)
        )  # B[1]
        self.B.append(
            tf.constant([[sys_params['alpha1'] * sys_params['k1'], 0],
                         [0, sys_params['alpha2'] * sys_params['k2']],
                         [0, (1 - sys_params['alpha2']) * sys_params['k2']],
                         [(1 - sys_params['alpha1']) * sys_params['k1'], 0]], dtype=tf.float32)
        )  # B[2]

        self.two_g_sqrt = tf.sqrt(tf.constant(2 * sys_params['g'], dtype=tf.float32))

        # Data normalizers
        self.t_normalizer = t_normalizer
        self.v_normalizer = v_normalizer
        self.h_normalizer = h_normalizer

        if self.t_normalizer is None or self.v_normalizer is None or self.h_normalizer is None:
            self.data_is_normalized = False
        else:
            self.data_is_normalized = True

        # Initialize NN
        self.layers = [7] + hidden_layers + [4]
        self.weights, self.biases = self.initialize_nn(self.layers)
        self.initial_weights, self.initial_biases = copy.deepcopy(self.weights), copy.deepcopy(self.biases)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        # Training losses
        self.train_f_loss = []
        self.train_u_loss = []
        self.train_total_loss = []

        # Validation loss
        self.validation_loss = []

    def initialize_nn(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(1, num_layers):
            W = self.xavier_init(size=[layers[l], layers[l - 1]])
            b = tf.Variable(tf.zeros([layers[l], 1]), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def predict(self, np_prediction_t, np_prediction_v, np_prediction_ic):
        tf_x = tf.constant(np.concatenate([np_prediction_t,
                                           np_prediction_v,
                                           np_prediction_ic], axis=0), dtype=tf.float32)

        return self.nn(tf_x).numpy()

    def nn(self, tf_x):
        num_layers = len(self.weights) + 1
        tf_U = tf_x
        for l in range(0, num_layers - 2):
            tf_W = self.weights[l]
            tf_b = self.biases[l]
            tf_U = tf.tanh(tf.add(tf.matmul(tf_W, tf_U), tf_b))
        tf_W = self.weights[-1]
        tf_b = self.biases[-1]
        tf_Y = tf.add(tf.matmul(tf_W, tf_U), tf_b)

        return tf_Y

    def f(self, tf_x, tf_v):
        # ODE sys: dh1_dt = -(a1/A1)*sqrt(2*g*h1) + (a3/A1)*sqrt(2*g*h3) + ((alpha1*k1)/A1)*v1
        #          dh2_dt = -(a2/A2)*sqrt(2*g*h2) + (a4/A2)*sqrt(2*g*h4) + ((alpha2*k2)/A2)*v2
        #          dh3_dt = -(a3/A3)*sqrt(2*g*h3) + (((1 - alpha2)*k2)/A3)*v2
        #          dh4_dt = -(a4/A4)*sqrt(2*g*h4) + (((1 - alpha1)*k1)/A4)*v1
        #
        # In matrix form: B[0]*dot_H + sqrt(2*g)*B[1]*sqrt(H) - B[2]*V

        tf_nn1_selector = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        tf_nn2_selector = tf.constant([[0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        tf_nn3_selector = tf.constant([[0.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
        tf_nn4_selector = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gtf:
            gtf.watch(tf_x)
            tf_nn = self.nn(tf_x)
            tf_nn1 = tf.matmul(tf_nn1_selector, tf_nn)
            tf_nn2 = tf.matmul(tf_nn2_selector, tf_nn)
            tf_nn3 = tf.matmul(tf_nn3_selector, tf_nn)
            tf_nn4 = tf.matmul(tf_nn4_selector, tf_nn)
        tf_dnn1_dx = gtf.gradient(tf_nn1, tf_x)
        tf_dnn2_dx = gtf.gradient(tf_nn2, tf_x)
        tf_dnn3_dx = gtf.gradient(tf_nn3, tf_x)
        tf_dnn4_dx = gtf.gradient(tf_nn4, tf_x)
        tf_dnn_dt = tf.concat([tf.slice(tf_dnn1_dx, [0, 0], [1, tf_dnn1_dx.shape[1]]),
                               tf.slice(tf_dnn2_dx, [0, 0], [1, tf_dnn2_dx.shape[1]]),
                               tf.slice(tf_dnn3_dx, [0, 0], [1, tf_dnn3_dx.shape[1]]),
                               tf.slice(tf_dnn4_dx, [0, 0], [1, tf_dnn4_dx.shape[1]])], axis=0)
        if self.data_is_normalized:
            return (self.h_normalizer.std / self.t_normalizer.std) * tf.matmul(self.B[0], tf_dnn_dt) + \
                   self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(self.h_normalizer.denormalize(tf_nn))) - \
                   tf.matmul(self.B[2], self.v_normalizer.denormalize(tf_v))
        else:
            return tf.matmul(self.B[0], tf_dnn_dt) + \
                   self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(tf_nn)) - \
                   tf.matmul(self.B[2], tf_v)

    def train(self, np_train_u_t, np_train_u_v, np_train_u_ic, np_train_f_t, np_train_f_v, np_train_f_ic,
              np_validation_t, np_validation_v, np_validation_ic, np_validation_h, max_epochs=10000, stop_loss=0.0005):
        # Train data
        tf_train_u_x = tf.constant(np.concatenate([np_train_u_t,
                                                   np_train_u_v,
                                                   np_train_u_ic], axis=0), dtype=tf.float32)
        tf_train_u_ic = tf.constant(np_train_u_ic, dtype=tf.float32)

        np_train_f_x = np.concatenate([
            np_train_f_t,
            np_train_f_v,
            np_train_f_ic], axis=0)
        np.random.shuffle(np.transpose(np_train_f_x))

        tf_train_f_x = tf.constant(np_train_f_x, dtype=tf.float32)
        tf_train_f_v = tf.constant(np_train_f_x[1:3], dtype=tf.float32)

        # Validation data
        tf_val_x = tf.constant(np.concatenate([np_validation_t,
                                               np_validation_v,
                                               np_validation_ic], axis=0), dtype=tf.float32)
        tf_val_h = tf.constant(np_validation_h, dtype=tf.float32)

        # Training process
        epoch = 0
        tf_val_loss = tf.constant(np.Inf, dtype=tf.float32)
        np_val_loss = tf_val_loss.numpy()
        tf_best_val_loss = copy.deepcopy(tf_val_loss)
        epochs_over_analysis = 100
        # val_moving_average_queue = Queue(maxsize=epochs_over_analysis) # TODO: Improve validation loss analysis
        # last_val_moving_average = tf_val_total_loss.numpy()
        best_weights = copy.deepcopy(self.weights)
        best_biases = copy.deepcopy(self.biases)
        loss_rising = False
        while epoch < max_epochs and tf_val_loss > stop_loss and not loss_rising:
            # Learning rate adjustments
            if epoch % 10000 == 0:
                self.learning_rate = self.learning_rate / 2
                self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999,
                                                    epsilon=1e-07)

            # Gradients
            grad_weights, grad_biases = self.get_grads(tf_train_u_x, tf_train_u_ic, tf_train_f_x, tf_train_f_v,
                                                       f_loss_weight=0.1)

            # Updating weights and biases
            grads = grad_weights + grad_biases
            vars_to_update = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars_to_update))

            # Validation
            tf_val_nn = self.nn(tf_val_x)
            tf_val_loss = tf.reduce_mean(tf.square(tf_val_nn - tf_val_h))
            np_val_loss = tf_val_loss.numpy()
            self.validation_loss.append(np_val_loss)

            if tf_val_loss < tf_best_val_loss:
                tf_best_val_loss = copy.deepcopy(tf_val_loss)
                best_weights = copy.deepcopy(self.weights)
                best_biases = copy.deepcopy(self.biases)

            # if val_moving_average_queue.full():  # TODO: Improve validation loss analysis
            #     val_moving_average_queue.get()
            # val_moving_average_queue.put(np_val_loss)
            #
            if epoch % epochs_over_analysis == 0:
                print('Validation loss on epoch ' + str(epoch) + ': ' + str(np_val_loss))
            #
            #     val_moving_average = sum(val_moving_average_queue.queue) / val_moving_average_queue.qsize()
            #     if val_moving_average > last_val_moving_average:
            #         loss_rising = True
            #     else:
            #         last_val_moving_average = val_moving_average

            epoch = epoch + 1
        self.weights = best_weights
        self.biases = best_biases
        print('Validation loss at the training\'s end: ' + str(np_val_loss))

    def get_grads(self, tf_u_x, tf_u_ic, tf_f_x, tf_f_v, u_loss_weight=1.0, f_loss_weight=1.0):
        with tf.GradientTape(persistent=True) as gtu:
            tf_u_nn = self.nn(tf_u_x)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_nn - tf_u_ic))
            self.train_u_loss.append(tf_u_loss.numpy())

            tf_f_nn = self.f(tf_f_x, tf_f_v)
            tf_f_loss = tf.reduce_mean(tf.square(tf_f_nn))
            self.train_f_loss.append(tf_f_loss.numpy())

            tf_total_loss = u_loss_weight * tf_u_loss + f_loss_weight * tf_f_loss
            self.train_total_loss.append(tf_total_loss.numpy())
        grad_weights = gtu.gradient(tf_total_loss, self.weights)
        grad_biases = gtu.gradient(tf_total_loss, self.biases)

        return grad_weights, grad_biases

    def reset_nn(self):
        self.weights = copy.deepcopy(self.initial_weights)
        self.biases = copy.deepcopy(self.initial_biases)
