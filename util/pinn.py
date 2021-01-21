import os
import copy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from util.factory_lbfgs import function_factory
from util.normalizer import Normalizer
from util.data_interface import JsonDAO


class PINN:
    """Physics informed neural network main class"""

    def __init__(self, n_inputs, n_outputs, hidden_layers, units_per_layer,
                 X_normalizer=None, Y_normalizer=None, learning_rate=0.001, random_seed=None):
        """
        It initializes some PINN parameters regarding to the network structure mainly.

        :param n_inputs: Number of neural network inputs
        :type n_inputs: int
        :param n_outputs: Number of neural network outputs
        :type n_outputs: int
        :param hidden_layers: Number of hidden layers
        :type hidden_layers: int
        :param units_per_layer: Number of neurons in each hidden layer
        :type units_per_layer: int
        :param X_normalizer: Object responsible for the neural network input normalization
            (default is None)
        :type X_normalizer: util.normalizer.Normalizer
        :param Y_normalizer: Object responsible for the neural network output denormalization
            (default is None)
        :type Y_normalizer: util.normalizer.Normalizer
        :param learning_rate: Learning rate
            (default is 0.001)
        :type learning_rate: float
        :param random_seed: Random seed for weight and bias initialization
            (default is None)
        :type random_seed: int
        """

        # Neural network structure
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer

        # Normalizers
        self.X_normalizer = X_normalizer
        self.Y_normalizer = Y_normalizer

        # Model
        self.model = None
        if X_normalizer is not None and Y_normalizer is not None:
            self.model_init(random_seed)

        # Trained T
        self.trained_T = None

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

        # JSON interface
        self.dao = JsonDAO()

    def model_init(self, random_seed=None):
        """
        It initializes a Keras model based on the class attributes.

        :param random_seed: Random seed for weight and bias generation
            (default is None)
        :type random_seed: int
        """

        if self.X_normalizer is None or self.Y_normalizer is None:
            raise Exception('Before initializing the neural network, the class normalizers must be defined.')
        else:
            tf.keras.backend.set_floatx('float64')
            if random_seed is not None:
                np.random.seed(random_seed)
                tf.random.set_seed(random_seed)
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.Input(shape=(self.n_inputs,)))
            # Normalizer
            self.model.add(tf.keras.layers.Lambda(lambda tf_X: self.X_normalizer.normalize(tf_X)))
            for i in range(self.hidden_layers):
                self.model.add(tf.keras.layers.Dense(self.units_per_layer, 'tanh', kernel_initializer='glorot_normal'))
            self.model.add(tf.keras.layers.Dense(self.n_outputs, None, kernel_initializer='glorot_normal'))

    def predict(self, np_X, np_ic=None, prediction_T=None, return_raw=False, time_column=0):
        """
        It feeds Keras model with the input X and returns its prediction. If X's time column contains t > T and the
        initial conditions are given, the function split and processes X based on the given T before feeding the neural
        network. Otherwise, the function expects X to contain the initial condition and t < trained T in each row. The
        function also makes possible to put multiple simulations [0, tf] with tf > T in a single X input.

        :param np_X: Input X
        :type np_X: numpy.ndarray
        :param np_ic: Initial conditions
            (default is None)
        :type np_ic: numpy.ndarray
        :param prediction_T: Prediction period
            (default is None)
        :type prediction_T: float
        :param return_raw: Sets the function to return a Tensor instead of a ndarray
            (default is False)
        :type return_raw: bool
        :param time_column: Time column in X
            (default is 0)
        :type time_column: int
        :returns: Neural network's response
        :rtype: numpy.ndarray or tensorflow.Tensor
        """

        if np_ic is None:
            tf_X = self.tensor(np_X)
            tf_NN = self.model(tf_X)

            if return_raw:
                return tf_NN
            else:
                return self.Y_normalizer.denormalize(tf_NN.numpy())
        elif np_X.shape[1] + np_ic.shape[1] == self.n_inputs:
            if prediction_T is not None:
                np_Z = self.process_input(np_X, np_ic, prediction_T, time_column)
                tf_X = self.tensor(np_Z)
                tf_NN = self.model(tf_X)

                if return_raw:
                    return tf_NN
                else:
                    return self.Y_normalizer.denormalize(tf_NN.numpy())
            else:
                raise Exception('Missing neural network working period.')
        else:
            raise Exception('np_X dimension plus np_ic dimension do not match neural network input dimension')

    def process_input(self, np_X, np_ic, prediction_T, time_column):
        """
        It processes the input X, which contains time instants bigger than the minimum between neural network's training
        T and the given prediction T, to shape it for feeding directly into the neural network. The function also works
        for multiple simulations [0, tf] with tf > T in a single X input.

        :param np_X: Input X
        :type np_X: numpy.ndarray
        :param np_ic: Initial conditions
            (default is None)
        :type np_ic: numpy.ndarray
        :param prediction_T: Prediction period
            (default is None)
        :type prediction_T: float
        :param time_column: Time column in X
            (default is 0)
        :type time_column: int
        :returns: Processed X
        :rtype: numpy.ndarray
        """

        # Trained T checking
        if self.trained_T is None:
            raise Exception('The parameter "trained_T" must be set before a long signal prediction.')

        # Detection of different simulations
        simulation_indexes = np.where(np_X[:, time_column] == 0.0)[0].tolist()
        simulation_indexes.append(np_X.shape[0])

        # Processing each simulation
        res = []
        min_T = min(prediction_T, self.trained_T)
        for k in range(len(simulation_indexes) - 1):
            np_Z = copy.deepcopy(np_X[simulation_indexes[k]:simulation_indexes[k+1], :])

            # Filling spaces bigger than T by inserting some extra samples
            if prediction_T > self.trained_T:
                previous_t = np_Z[0, time_column]
                inserted_lines = []
                i = 1
                while i < np_Z.shape[0]:
                    if np_Z[i, time_column] - previous_t > self.trained_T:
                        np_line = copy.deepcopy(np_Z[i - 1, :])
                        np_line[time_column] = previous_t + self.trained_T/2
                        np_Z = np.insert(np_Z, i, np_line, axis=0)
                        inserted_lines.append(i)
                    previous_t = np_Z[i, time_column]
                    i = i + 1

            # Rewriting of time values to fit them in T
            np_Z[:, time_column] = np_Z[:, time_column] % min_T

            # Initial conditions calculus
            previous_t = np_Z[0, time_column]
            np_y0 = np.reshape(np_ic[k, :], (1, np_ic[k, :].size))
            new_columns = [np_y0]
            for i in range(1, np_Z.shape[0]):
                row = np_Z[i]
                if row[time_column] <= previous_t:
                    np_w = copy.deepcopy(np_Z[i - 1, :])
                    np_w[time_column] = min_T
                    prediction_input = np.array([np.append(np_w, np_y0)])
                    np_y0 = self.predict(prediction_input)
                previous_t = row[time_column]
                new_columns.append(np_y0)

            # Merging of time/control inputs and initial conditions
            np_new_columns = np.concatenate(new_columns)
            if len(np_new_columns.shape) == 1:
                np_new_columns = np.reshape(np_new_columns, (np_new_columns.shape[0], 1))

            np_sim_res = np.append(np_Z, np_new_columns, axis=1)

            # Deleting inserted data
            if prediction_T > self.trained_T:
                np_sim_res = np.delete(np_sim_res, inserted_lines, axis=0)

            # Saving simulation processed data
            res.append(np_sim_res)

        return np.concatenate(res)

    def tensor(self, np_X):
        """
        It converts a ndarray into a Tensor.

        :param np_X: Numpy X
        :type np_X: numpy.ndarray or list or float
        :returns: Tensorflow X
        :rtype: tensorflow.Tensor
        """

        return tf.convert_to_tensor(np_X, dtype=tf.dtypes.float64)

    def set_opt_params(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """
        It sets Adam optimizer's parameters.

        :param learning_rate: Learning rate
        :type learning_rate: float
        :param beta_1: Beta 1
        :type beta_1: float
        :param beta_2: Beta 2
        :type beta_2: float
        :param epsilon: Epsilon
        :type epsilon: float
        """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1,
                                                  beta_2=beta_2, epsilon=epsilon)

    def train(self, np_train_u_X, np_train_u_Y, np_train_f_X, train_T, np_val_X, np_val_ic, val_T, np_val_Y,
              adam_epochs=500, max_lbfgs_iterations=1000, epochs_per_print=100,
              u_loss_weight=1.0, f_loss_weight=0.1, save_losses=True):
        """
        It trains the neural network using Adam and L-BFGS algorithms. It does not split training data into
        mini-batches.

        :param np_train_u_X: MSEu inputs
        :type np_train_u_X: numpy.ndarray
        :param np_train_u_Y: MSEu labels
        :type np_train_u_Y: numpy.ndarray
        :param np_train_f_X: MSEf inputs
        :type np_train_f_X: numpy.ndarray
        :param train_T: Max t the neural network is going to be able to predict
        :type train_T: float
        :param np_val_X: Validation inputs
        :type np_val_X: numpy.ndarray
        :param np_val_ic: Validation initial conditions
        :type np_val_ic: numpy.ndarray
        :param val_T: Validation T
        :type val_T: float
        :param np_val_Y: Validation labels
        :type np_val_Y: numpy.ndarray
        :param adam_epochs: Epochs to train with Adam
            (default is 500)
        :type adam_epochs: int
        :param max_lbfgs_iterations: Iterations to train with L-BFGS
            (default is 1000)
        :type max_lbfgs_iterations: int
        :param epochs_per_print: The function prints the total MSE each time epochs % epochs_per_print = 0
            (default is 100)
        :type epochs_per_print: int
        :param u_loss_weight: MSEu weight
            (default is 1.0)
        :type u_loss_weight: float
        :param f_loss_weight: MSEf weight
            (default is 0.1)
        :type f_loss_weight: float
        :param save_losses: To save or not the losses in the class attributes
            (default is True)
        :type save_losses: bool
        """

        # Trained T parameter update
        self.trained_T = train_T

        # Numpy data to tensorflow data
        tf_train_u_X = self.tensor(np_train_u_X)
        tf_train_u_Y = self.tensor(self.Y_normalizer.normalize(np_train_u_Y))
        tf_train_f_X = self.tensor(np_train_f_X)

        # Validation normalized labels
        tf_val_Y = self.tensor(self.Y_normalizer.normalize(np_val_Y))

        # Training with Adam
        self.train_adam(tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                        adam_epochs, epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

        # Training with L-BFGS
        self.train_lbfgs(tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                         max_lbfgs_iterations, epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

    def train_adam(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                   epochs, epochs_per_print, u_loss_weight, f_loss_weight, save_losses):
        """
        It trains the neural network using Adam algorithm. It does not split training data into mini-batches.

        :param tf_train_u_X: MSEu inputs
        :type tf_train_u_X: tensorflow.Tensor
        :param tf_train_u_Y: MSEu labels
        :type tf_train_u_Y: tensorflow.tensor
        :param tf_train_f_X: MSEf inputs
        :type tf_train_f_X: tensorflow.Tensor
        :param np_val_X: Validation inputs
        :type np_val_X: numpy.ndarray
        :param np_val_ic: Validation initial conditions
        :type np_val_ic: numpy.ndarray
        :param val_T: Validation T
        :type val_T: float
        :param tf_val_Y: Validation labels
        :type tf_val_Y: tensorflow.Tensor
        :param epochs: Epochs to train
        :type epochs: int
        :param epochs_per_print: The function prints the total MSE each time epochs % epochs_per_print = 0
        :type epochs_per_print: int
        :param u_loss_weight: MSEu weight
        :type u_loss_weight: float
        :param f_loss_weight: MSEf weight
        :type f_loss_weight: float
        :param save_losses: To save or not the losses in the class attributes
        :type save_losses: bool
        """

        # Training states and variables
        epoch = 0
        grads = None

        tf_val_loss = self.tensor(np.Inf)
        tf_best_val_loss = copy.deepcopy(tf_val_loss)

        best_weights = copy.deepcopy(self.model.get_weights())

        # Training process
        while epoch <= epochs:
            # Updating weights and biases
            if grads is not None:
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Learning rate adjustments
            if epoch % 100 == 0:
                self.learning_rate = self.learning_rate / 2
                self.set_opt_params(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

            # Network gradients
            with tf.GradientTape(persistent=True) as tape:
                tf_total_loss, tf_u_loss, tf_f_loss = self.get_losses(tf_train_u_X, tf_train_u_Y, tf_train_f_X,
                                                                      u_loss_weight, f_loss_weight)
            grads = tape.gradient(tf_total_loss, self.model.trainable_variables)

            # Validation
            tf_val_NN = self.predict(np_val_X, np_val_ic, val_T, return_raw=True)
            tf_val_loss = tf.reduce_mean(tf.square(tf_val_NN - tf_val_Y))
            np_val_loss = tf_val_loss.numpy()

            if tf_val_loss < tf_best_val_loss:
                tf_best_val_loss = copy.deepcopy(tf_val_loss)
                best_weights = copy.deepcopy(self.model.get_weights())

            # Saving loss values
            if save_losses:
                self.train_total_loss.append(tf_total_loss.numpy())
                self.train_u_loss.append(tf_u_loss.numpy())
                self.train_f_loss.append(tf_f_loss.numpy())

                self.validation_loss.append(np_val_loss)

            if epoch % epochs_per_print == 0:
                print('Epoch:', str(epoch), '-', 'Adam validation loss:', str(np_val_loss))

            # Epoch count
            epoch = epoch + 1

        # Epoch adjustment
        epoch = epoch - 1

        # Updating the neural network with the best weights
        self.model.set_weights(best_weights)

        # Final validation loss printing
        print('Validation loss at the end of Adam -> Epoch:', str(epoch), '-',
              'validation loss:', tf_best_val_loss.numpy())

    def get_losses(self, tf_u_X, tf_u_Y, tf_f_X, u_loss_weight, f_loss_weight):
        """
        It returns the MSE, MSEu and MSEf values for the given inputs.

        :param tf_u_X: MSEu inputs
        :type tf_u_X: tensorflow.Tensor
        :param tf_u_Y: MSEu labels
        :type tf_u_Y: tensorflow.Tensor
        :param tf_f_X: MSEf inputs
        :type tf_f_X: tensorflow.Tensor
        :param u_loss_weight: MSEu weight
        :type u_loss_weight: float
        :param f_loss_weight: MSEf weight
        :type f_loss_weight: float
        :returns: MSE, MSEu and MSEf
        :rtype: tuple
        """

        # MSEu
        tf_u_NN = self.model(tf_u_X)
        tf_u_loss = tf.reduce_mean(tf.square(tf_u_NN - tf_u_Y))

        # MSEf
        tf_f_NN = self.f(tf_f_X)
        tf_f_loss = tf.reduce_mean(tf.square(tf_f_NN))

        # MSE
        tf_weighted_u_loss = u_loss_weight * tf_u_loss
        tf_weighted_f_loss = f_loss_weight * tf_f_loss
        tf_total_loss = tf_weighted_u_loss + tf_weighted_f_loss

        return tf_total_loss, tf_weighted_u_loss, tf_weighted_f_loss

    def f(self, tf_X):
        """
        It computes the physics informed function f(X) for minimization.

        :param tf_X: MSEf inputs
        :type tf_X: tensorflow.Tensor
        :returns: f(X)
        :rtype: tensorflow.Tensor
        """

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as f_tape:
            f_tape.watch(tf_X)

            # Neural network output calculus
            tf_NN = self.Y_normalizer.denormalize(self.model(tf_X))

            # Output decomposition
            np_output_selector = np.eye(self.n_outputs)
            decomposed_NN = []
            for i in range(self.n_outputs):
                tf_NN_single_output = tf.matmul(tf_NN, tf.transpose(self.tensor([np_output_selector[i]])))
                decomposed_NN.append(tf_NN_single_output)

        return self.expression(tf_X, tf_NN, decomposed_NN, f_tape)

    def train_lbfgs(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                    max_iterations, epochs_per_print, u_loss_weight, f_loss_weight, save_losses):
        """
        It trains the neural network using Adam algorithm. It does not split training data into mini-batches.

        :param tf_train_u_X: MSEu inputs
        :type tf_train_u_X: tensorflow.Tensor
        :param tf_train_u_Y: MSEu labels
        :type tf_train_u_Y: tensorflow.tensor
        :param tf_train_f_X: MSEf inputs
        :type tf_train_f_X: tensorflow.Tensor
        :param np_val_X: Validation inputs
        :type np_val_X: numpy.ndarray
        :param np_val_ic: Validation initial conditions
        :type np_val_ic: numpy.ndarray
        :param val_T: Validation T
        :type val_T: float
        :param tf_val_Y: Validation labels
        :type tf_val_Y: tensorflow.Tensor
        :param max_iterations: Max iterations to train
        :type max_iterations: int
        :param epochs_per_print: The function prints the total MSE each time epochs % epochs_per_print = 0
        :type epochs_per_print: int
        :param u_loss_weight: MSEu weight
        :type u_loss_weight: float
        :param f_loss_weight: MSEf weight
        :type f_loss_weight: float
        :param save_losses: To save or not the losses in the class attributes
        :type save_losses: bool
        """

        func = function_factory(self, tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                                epochs_per_print, u_loss_weight, f_loss_weight, save_losses)

        # Converting initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
        res = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params,
                                           max_iterations=max_iterations)  # Each iteration is equivalent to 2-4 epochs

        # After training, the final optimized parameters are still in res.position
        # So we have to manually put them back to the model
        func.assign_new_model_parameters(res.position)

    def save(self, directory_path):
        """
        It saves the training losses, the normalizers' attributes and the neural network weights in the given
        directory (creating it if necessary).

        :param directory_path: Directory path
        :type directory_path: str
        """

        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        self.save_losses(directory_path + '/losses.json')
        self.save_normalizers(directory_path + '/normalizers.json')
        self.save_weights(directory_path + '/weights.h5')

    def save_losses(self, file_path):
        """
        It saves the training losses in a JSON file.

        :param file_path: File path
        :type file_path: str
        """

        losses = {'train_u_loss': self.train_u_loss,
                  'train_f_loss': self.train_f_loss,
                  'train_total_loss': self.train_total_loss,
                  'validation_loss': self.validation_loss}
        self.dao.save(file_path, losses)

    def save_normalizers(self, file_path):
        """
        It saves the normalizers' attributes in a JSON file.

        :param file_path: File path
        :type file_path: str
        """

        X_normalizer = copy.deepcopy(self.X_normalizer)
        X_normalizer.mean = X_normalizer.mean.tolist()
        X_normalizer.std = X_normalizer.std.tolist()

        Y_normalizer = copy.deepcopy(self.Y_normalizer)
        Y_normalizer.mean = Y_normalizer.mean.tolist()
        Y_normalizer.std = Y_normalizer.std.tolist()

        normalizers = {'X_normalizer': X_normalizer.__dict__,
                       'Y_normalizer': Y_normalizer.__dict__}
        self.dao.save(file_path, normalizers)

    def save_weights(self, file_path):
        """
        It saves the neural network weights and biases in a H5 file.

        :param file_path: File path
        :type file_path: str
        """

        self.model.save_weights(file_path)

    def load(self, directory_path):
        """
        It loads the training losses, the normalizers' attributes and the neural network weights in the given
        directory.

        :param directory_path: Directory path
        :type directory_path: str
        """

        self.load_losses(directory_path + '/losses.json')
        self.load_normalizers(directory_path + '/normalizers.json')
        self.load_weights(directory_path + '/weights.h5')

    def load_losses(self, file_path):
        """
        It loads the training losses from the given JSON file.

        :param file_path: File path
        :type file_path: str
        """

        losses = self.dao.load(file_path)
        self.train_u_loss = losses['train_u_loss']
        self.train_f_loss = losses['train_f_loss']
        self.train_total_loss = losses['train_total_loss']
        self.validation_loss = losses['validation_loss']

    def load_normalizers(self, file_path):
        """
        It loads the normalizers' attributes from the given JSON file.

        :param file_path: File path
        :type file_path: str
        """

        normalizers = self.dao.load(file_path)
        X_normalizer = Normalizer(normalizers['X_normalizer'])
        Y_normalizer = Normalizer(normalizers['Y_normalizer'])

        self.X_normalizer = X_normalizer
        self.Y_normalizer = Y_normalizer

        self.model_init()

    def load_weights(self, file_path):
        """
        It loads the neural network weights and biases from the given H5 file.

        :param file_path: File path
        :type file_path: str
        """

        self.model.load_weights(file_path)

    def get_weights(self):
        """
        Returns the neural network weights and biases in a list.

        :returns: Neural network weights and biases
        :rtype: list
        """

        weights = []
        for layer in self.model.layers:
            weights.append(layer.get_weights())

        return weights

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        """
        The one sided ODE or PDE expression for the neural network physics informing. This is the function the user must
        overwrite for each system.

        :param tf_X: Neural network inputs
        :type tf_X: tensorflow.Tensor
        :param tf_NN: Neural network outputs
        :type tf_X: tensorflow.Tensor
        :param decomposed_NN: A list with each output as a vector
        :type decomposed_NN: list
        :param tape: Object used in the automatic differentiation
        :type tape: tensorflow.GradientTape
        :returns: f value
        :rtype: tensorflow.Tensor
        """

        return self.tensor(0.0)


class OneTankPINN(PINN):
    """Physics informed neural network for the one tank system"""

    def __init__(self, sys_params, hidden_layers, units_per_layer,
                 X_normalizer=None, Y_normalizer=None, learning_rate=0.001, random_seed=None):
        """
        It initializes some PINN parameters regarding to the network structure mainly.

        :param sys_params: System parameters. Structure:
            {
                'g': g,  # [cm/s^2]
                'a': a,  # [cm^2]
                'A': A,  # [cm^2]
                'k': k   # [cm^3/Vs]
            }
        :type sys_params: dict
        :param hidden_layers: Number of hidden layers
        :type hidden_layers: int
        :param units_per_layer: Number of neurons in each hidden layer
        :type units_per_layer: int
        :param X_normalizer: Object responsible for the neural network input normalization
            (default is None)
        :type X_normalizer: util.normalizer.Normalizer
        :param Y_normalizer: Object responsible for the neural network output denormalization
            (default is None)
        :type Y_normalizer: util.normalizer.Normalizer
        :param learning_rate: Learning rate
            (default is 0.001)
        :type learning_rate: float
        :param random_seed: Random seed for weight and bias initialization
            (default is None)
        :type random_seed: int
        """

        super().__init__(3, 1, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate, random_seed)

        self.k = sys_params['k']
        self.a = sys_params['a']
        self.A = sys_params['A']
        self.two_g_sqrt = tf.sqrt(self.tensor(2 * sys_params['g']))

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        """
        The one sided ODE expression for the one tank system.
        ODE: dh_dt = (k/A)*v - (a/A)*sqrt(2*g*h)

        :param tf_X: Neural network inputs
        :type tf_X: tensorflow.Tensor
        :param tf_NN: Neural network outputs
        :type tf_X: tensorflow.Tensor
        :param decomposed_NN: A list with each output as a vector
        :type decomposed_NN: list
        :param tape: Object used in the automatic differentiation
        :type tape: tensorflow.GradientTape
        :returns: f value
        :rtype: tensorflow.Tensor
        """

        tf_v = tf.slice(tf_X, [0, 1], [tf_X.shape[0], 1])

        tf_dnn_dx = tape.gradient(tf_NN, tf_X)
        tf_dnn_dt = tf.slice(tf_dnn_dx, [0, 0], [tf_dnn_dx.shape[0], 1])

        return self.A * tf_dnn_dt + (self.a * self.two_g_sqrt) * tf.sqrt(tf.maximum(tf_NN, 0.0)) - self.k * tf_v


class VanDerPolPINN(PINN):
    """Physics informed neural network for the Van der Pol oscillator"""

    def __init__(self, hidden_layers, units_per_layer,
                 X_normalizer=None, Y_normalizer=None, learning_rate=0.001, random_seed=None):
        """
        It initializes some PINN parameters regarding to the network structure mainly.

        :param hidden_layers: Number of hidden layers
        :type hidden_layers: int
        :param units_per_layer: Number of neurons in each hidden layer
        :type units_per_layer: int
        :param X_normalizer: Object responsible for the neural network input normalization
            (default is None)
        :type X_normalizer: util.normalizer.Normalizer
        :param Y_normalizer: Object responsible for the neural network output denormalization
            (default is None)
        :type Y_normalizer: util.normalizer.Normalizer
        :param learning_rate: Learning rate
            (default is 0.001)
        :type learning_rate: float
        :param random_seed: Random seed for weight and bias initialization
            (default is None)
        :type random_seed: int
        """

        super().__init__(4, 2, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate, random_seed)

        # System parameters for matrix form
        self.A = self.tensor([[1, 1],
                              [-1, 0]])
        self.b = self.tensor([[1, 0]])

    def expression(self, tf_X, tf_NN, decomposed_NN, tape):
        """
        The one sided ODE expression for the Van der Pol oscillator.
        ODE: d2x_dt2 = (1 - x^2) * dx_dt - x + u

        As 2 states system: dx1_dt = (1 - x2^2) * x1 - x2 + u
                            dx2_dt = x1

        (x2 is the position variable)

        In matrix form: dX_dt = X*A + (u - x2^2 * x1)*b

        :param tf_X: Neural network inputs
        :type tf_X: tensorflow.Tensor
        :param tf_NN: Neural network outputs
        :type tf_X: tensorflow.Tensor
        :param decomposed_NN: A list with each output as a vector
        :type decomposed_NN: list
        :param tape: Object used in the automatic differentiation
        :type tape: tensorflow.GradientTape
        :returns: f value
        :rtype: tensorflow.Tensor
        """

        tf_u = tf.slice(tf_X, [0, 1], [tf_X.shape[0], 1])

        tf_dnn1_dx = tape.gradient(decomposed_NN[0], tf_X)
        tf_dnn2_dx = tape.gradient(decomposed_NN[1], tf_X)

        tf_dnn_dt = tf.concat([tf.slice(tf_dnn1_dx, [0, 0], [tf_dnn1_dx.shape[0], 1]),
                               tf.slice(tf_dnn2_dx, [0, 0], [tf_dnn2_dx.shape[0], 1])], axis=1)

        return tf_dnn_dt - tf.matmul(tf_NN, self.A) + \
               tf.matmul((tf.square(decomposed_NN[1]) * decomposed_NN[0] - tf_u), self.b)


class FourTanksPINN(PINN):
    """Physics informed neural network for the four tanks system"""

    def __init__(self, sys_params, hidden_layers, units_per_layer,
                 X_normalizer=None, Y_normalizer=None, learning_rate=0.001, random_seed=None):
        """
        It initializes some PINN parameters regarding to the network structure mainly.

        :param sys_params: System parameters. Structure:
            {
                'g': g,            # [cm/s^2]
                'a1': a1,          # [cm^2]
                'a2': a2,          # [cm^2]
                'a3': a3,          # [cm^2]
                'a4': a4,          # [cm^2]
                'A1': A1,          # [cm^2]
                'A2': A2,          # [cm^2]
                'A3': A3,          # [cm^2]
                'A4': A4,          # [cm^2]
                'alpha1': alpha1,  # [adm]
                'alpha2': alpha2,  # [adm]
                'k1': k1,          # [cm^3/Vs]
                'k2': k2,          # [cm^3/Vs]
            }
        :type sys_params: dict
        :param hidden_layers: Number of hidden layers
        :type hidden_layers: int
        :param units_per_layer: Number of neurons in each hidden layer
        :type units_per_layer: int
        :param X_normalizer: Object responsible for the neural network input normalization
            (default is None)
        :type X_normalizer: util.normalizer.Normalizer
        :param Y_normalizer: Object responsible for the neural network output denormalization
            (default is None)
        :type Y_normalizer: util.normalizer.Normalizer
        :param learning_rate: Learning rate
            (default is 0.001)
        :type learning_rate: float
        :param random_seed: Random seed for weight and bias initialization
            (default is None)
        :type random_seed: int
        """

        super().__init__(7, 4, hidden_layers, units_per_layer, X_normalizer, Y_normalizer, learning_rate, random_seed)

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
        """
        The one sided ODE expression for the four tanks system.
        ODE sys: dh1_dt = -(a1/A1)*sqrt(2*g*h1) + (a3/A1)*sqrt(2*g*h3) + ((alpha1*k1)/A1)*v1
                 dh2_dt = -(a2/A2)*sqrt(2*g*h2) + (a4/A2)*sqrt(2*g*h4) + ((alpha2*k2)/A2)*v2
                 dh3_dt = -(a3/A3)*sqrt(2*g*h3) + (((1 - alpha2)*k2)/A3)*v2
                 dh4_dt = -(a4/A4)*sqrt(2*g*h4) + (((1 - alpha1)*k1)/A4)*v1

        In matrix form: B[0]*dot_H + sqrt(2*g)*B[1]*sqrt(H) - B[2]*V

        :param tf_X: Neural network inputs
        :type tf_X: tensorflow.Tensor
        :param tf_NN: Neural network outputs
        :type tf_X: tensorflow.Tensor
        :param decomposed_NN: A list with each output as a vector
        :type decomposed_NN: list
        :param tape: Object used in the automatic differentiation
        :type tape: tensorflow.GradientTape
        :returns: f value
        :rtype: tensorflow.Tensor
        """

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

        tf_f_loss = tf.matmul(self.B[0], tf_dnn_dt) + \
                    self.two_g_sqrt * tf.matmul(self.B[1], tf.sqrt(tf.maximum(tf_nn, 0.0))) - tf.matmul(self.B[2], tf_v)

        return tf.transpose(tf_f_loss)
