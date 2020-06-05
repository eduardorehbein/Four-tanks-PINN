import copy
from queue import Queue
import tensorflow as tf
import numpy as np

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py


class PINN:
    def __init__(self, n_inputs, n_outputs, hidden_layers, units_per_layer, X_normalizer, Y_normalizer,
                 learning_rate=0.001, parallel_threads=8):
        # Parallel threads config
        # tf.config.threading.set_inter_op_parallelism_threads(parallel_threads)
        # tf.config.threading.set_intra_op_parallelism_threads(parallel_threads)

        # Input and output vectors' dimension
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Normalizers
        self.X_normalizer = X_normalizer
        self.Y_normalizer = Y_normalizer

        # Model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.n_inputs,)))
        self.model.add(tf.keras.layers.Lambda(lambda tf_X: self.X_normalizer.normalize(tf_X)))  # Normalize data
        for _ in range(hidden_layers):
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

    def predict(self, np_X):
        '''
        Predict the output u(t)
        :param np_X: data input points representing time t
        :return: u(t)
        '''

        tf_X = self.tensor(np_X)
        tf_NN = self.model(tf_X)
        return tf_NN.numpy()

    def tensor(self, np_X):
        return tf.convert_to_tensor(np_X, dtype=tf.dtypes.float32)

    def set_opt_params(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1,
                                                  beta_2=beta_2, epsilon=epsilon)

    def train(self, np_train_u_X, np_train_u_Y, np_train_f_X, np_val_X, np_val_Y,
              max_epochs=20000, stop_loss=0.0005, attach_loss=True):
        # Train data
        tf_train_u_X = self.tensor(np_train_u_X)
        tf_train_u_Y = self.tensor(np_train_u_Y)

        np_train_f_X = copy.deepcopy(np_train_f_X)
        np.random.shuffle(np_train_f_X)
        tf_train_f_X = self.tensor(np_train_f_X)

        # Validation data
        tf_val_X = self.tensor(np_val_X)
        tf_val_Y = self.tensor(np_val_Y)

        # Training process
        epoch = 0
        tf_val_loss = tf.constant(np.Inf, dtype=tf.float32)
        np_val_loss = tf_val_loss.numpy()
        tf_best_val_loss = copy.deepcopy(tf_val_loss)
        epochs_over_analysis = 100
        # val_moving_average_queue = Queue(maxsize=epochs_over_analysis)
        # last_val_moving_average = np_val_loss
        best_weights = copy.deepcopy(self.model.get_weights())
        loss_rising = False
        while epoch < max_epochs and tf_val_loss > stop_loss and not loss_rising:
            # Learning rate adjustments
            if epoch % 10000 == 0:
                self.learning_rate = self.learning_rate / 2
                self.set_opt_params(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

            # Updating weights and biases
            grads = self.get_grads(tf_train_u_X, tf_train_u_Y, tf_train_f_X,
                                   f_loss_weight=0.1, attach_losses=attach_loss)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Validation
            tf_val_NN = self.model(tf_val_X)
            tf_val_loss = tf.reduce_mean(tf.square(tf_val_NN - tf_val_Y))
            np_val_loss = tf_val_loss.numpy()
            if attach_loss:
                self.validation_loss.append(np_val_loss)

            if tf_val_loss < tf_best_val_loss:
                tf_best_val_loss = copy.deepcopy(tf_val_loss)
                best_weights = copy.deepcopy(self.model.get_weights())

            # if val_moving_average_queue.full():
            #     val_moving_average_queue.get()
            # val_moving_average_queue.put(np_val_loss)

            if epoch % epochs_over_analysis == 0:
                print('Validation loss on epoch ' + str(epoch) + ': ' + str(np_val_loss))

                # val_moving_average = sum(val_moving_average_queue.queue) / val_moving_average_queue.qsize()
                # if val_moving_average > last_val_moving_average:
                #     loss_rising = True
                # else:
                #     last_val_moving_average = val_moving_average

            epoch = epoch + 1
        self.model.set_weights(best_weights)
        print('Validation loss at the training\'s end: ' + str(np_val_loss))

    def get_grads(self, tf_u_X, tf_u_Y, tf_f_X, u_loss_weight=1.0, f_loss_weight=1.0, attach_losses=False):
        with tf.GradientTape(persistent=True) as total_tape:
            tf_u_NN = self.model(tf_u_X)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_NN - tf_u_Y))

            tf_f_NN = self.f(tf_f_X)
            tf_f_loss = tf.reduce_mean(tf.square(tf_f_NN))

            tf_weighted_u_loss = u_loss_weight * tf_u_loss
            tf_weighted_f_loss = f_loss_weight * tf_f_loss

            tf_total_loss = tf_weighted_u_loss + tf_weighted_f_loss
        grads = total_tape.gradient(tf_total_loss, self.model.trainable_variables)

        if attach_losses:
            self.train_u_loss.append(tf_weighted_u_loss.numpy())
            self.train_f_loss.append(tf_weighted_f_loss.numpy())
            self.train_total_loss.append(tf_total_loss.numpy())

        return grads

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

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        return self.tensor(0.0)


class FourTanksPINN(PINN):
    def __init__(self, sys_params, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate=0.001):
        super().__init__(7, 4, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate)

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
                 t_normalizer=None, v_normalizer=None, h_normalizer=None, parallel_threads=8):
        # Parallel threads config
        tf.config.threading.set_inter_op_parallelism_threads(parallel_threads)
        tf.config.threading.set_intra_op_parallelism_threads(parallel_threads)

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
