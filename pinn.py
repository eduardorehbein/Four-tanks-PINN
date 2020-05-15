import copy
from queue import Queue
import tensorflow as tf
import numpy as np

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py


class FourTanksPINN:
    def __init__(self, sys_params, hidden_layers, learning_rate, t_normalizer=None, v_normalizer=None, h_normalizer=None):
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
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        # Training loss
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
        tf_x = tf.constant(np.array([np_prediction_t, np_prediction_v, np_prediction_ic]), dtype=tf.float32)

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

        tf_selector = tf.eye(4)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gtf:
            gtf.watch(tf_x)
            tf_nn = self.nn(tf_x)
            tf_nn1 = tf.matmul(tf_selector[0], tf_nn)
            tf_nn2 = tf.matmul(tf_selector[1], tf_nn)
            tf_nn3 = tf.matmul(tf_selector[2], tf_nn)
            tf_nn4 = tf.matmul(tf_selector[3], tf_nn)
        tf_dnn1_dx = gtf.gradient(tf_nn1, tf_x)
        tf_dnn2_dx = gtf.gradient(tf_nn2, tf_x)
        tf_dnn3_dx = gtf.gradient(tf_nn3, tf_x)
        tf_dnn4_dx = gtf.gradient(tf_nn4, tf_x)
        tf_dnn_dt = tf.constant([tf_dnn1_dx[0],
                                 tf_dnn2_dx[0],
                                 tf_dnn3_dx[0],
                                 tf_dnn4_dx[0]], dtype=tf.float32)
        if self.data_is_normalized:
            return (self.h_normalizer.std / self.t_normalizer.std) * tf.matmul(self.B[0], tf_dnn_dt) + \
                   self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(self.h_normalizer.denormalize(tf_nn))) - \
                   tf.matmul(self.B[2], self.v_normalizer.denormalize(tf_v))
        else:
            return tf.matmul(self.B[0], tf_dnn_dt) + \
                   self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(tf_nn)) - \
                   tf.matmul(self.B[2], tf_v)

    def train(self, np_u_t, np_u_v, np_u_ic, np_f_t, np_f_v, np_f_ic, max_epochs=10000, stop_loss=0.0005):
        train_u_len = int(0.9 * len(np_u_t))
        train_f_len = int(0.9 * len(np_f_t))

        # Train data
        tf_train_u_x = tf.constant(
            np.array([
                np_u_t[:train_u_len],
                np_u_v[:train_u_len],
                np_u_ic[:train_u_len]]),
            dtype=tf.float32)
        tf_train_u_ic = tf.constant(np.array([np_u_ic[:train_u_len]]), dtype=tf.float32)

        np_train_f_x = np.array([
            np_f_t[:train_f_len],
            np_f_v[:train_f_len],
            np_f_ic[:train_f_len]
        ])
        np.random.shuffle(np.transpose(np_train_f_x))

        tf_train_f_x = tf.constant(np.array(np_train_f_x), dtype=tf.float32)
        tf_train_f_v = tf.constant(np.array(np_train_f_x[1:3]), dtype=tf.float32)

        # Validation data
        tf_val_u_x = tf.constant(
            np.array([
                np_u_t[train_u_len:],
                np_u_v[train_u_len:],
                np_u_ic[train_u_len:]]),
            dtype=tf.float32)
        tf_val_u_ic = tf.constant(np.array([np_u_ic[train_u_len:]]), dtype=tf.float32)

        np_val_f_x = np.array([
            np_f_t[train_f_len:],
            np_f_v[train_f_len:],
            np_f_ic[train_f_len:]
        ])
        np.random.shuffle(np.transpose(np_val_f_x))

        tf_val_f_x = tf.constant(np.array(np_val_f_x), dtype=tf.float32)
        tf_val_f_v = tf.constant(np.array(np_val_f_x[1:3]), dtype=tf.float32)

        # Training process
        epoch = 0
        tf_val_total_loss = tf.constant(1000, dtype=tf.float32)
        tf_val_best_total_loss = copy.deepcopy(tf_val_total_loss)
        val_moving_average_queue = Queue(maxsize=100)
        last_val_moving_average = tf_val_total_loss.numpy()
        best_weights = copy.deepcopy(self.weights)
        best_biases = copy.deepcopy(self.biases)
        loss_rising = False
        while epoch < max_epochs and tf_val_total_loss > stop_loss and not loss_rising:
            # Gradients
            grad_weights, grad_biases = self.get_grads(tf_train_u_x, tf_train_u_ic, tf_train_f_x, tf_train_f_v)

            # Updating weights and biases
            grads = grad_weights + grad_biases
            vars_to_update = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars_to_update))

            # Validation
            tf_val_u_predict = self.nn(tf_val_u_x)
            tf_val_u_loss = tf.reduce_mean(tf.square(tf_val_u_predict - tf_val_u_ic))

            tf_val_f_predict = self.f(tf_val_f_x, tf_val_f_v)
            tf_val_f_loss = tf.reduce_mean(tf.square(tf_val_f_predict))

            tf_val_total_loss = tf_val_u_loss + tf_val_f_loss

            if tf_val_total_loss < tf_val_best_total_loss:
                tf_val_best_total_loss = copy.deepcopy(tf_val_total_loss)
                best_weights = copy.deepcopy(self.weights)
                best_biases = copy.deepcopy(self.biases)

            if val_moving_average_queue.full():
                val_moving_average_queue.get()
            val_moving_average_queue.put(tf_val_total_loss.numpy())

            if epoch % 100 == 0:
                np_loss = tf_val_total_loss.numpy()
                self.validation_loss.append(np_loss)
                print('Validation loss on epoch ' + str(epoch) + ': ' + str(np_loss))

                val_moving_average = sum(val_moving_average_queue.queue) / val_moving_average_queue.qsize()
                if val_moving_average > last_val_moving_average:
                    loss_rising = True
                else:
                    last_val_moving_average = val_moving_average

            epoch = epoch + 1
        self.weights = best_weights
        self.biases = best_biases
        print('Validation loss at the training\'s end: ' + str(tf_val_total_loss.numpy()))

    def get_grads(self, tf_u_x, tf_u_ic, tf_f_x, tf_f_v):
        with tf.GradientTape(persistent=True) as gtu:
            tf_u_predict = self.nn(tf_u_x)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_predict - tf_u_ic))

            tf_f_predict = self.f(tf_f_x, tf_f_v)
            tf_f_loss = tf.reduce_mean(tf.square(tf_f_predict))

            tf_total_loss = tf_u_loss + tf_f_loss
        grad_weights = gtu.gradient(tf_total_loss, self.weights)
        grad_biases = gtu.gradient(tf_total_loss, self.biases)

        return grad_weights, grad_biases

    def reset_NN(self):
        self.weights = copy.deepcopy(self.initial_weights)
        self.biases = copy.deepcopy(self.initial_biases)
