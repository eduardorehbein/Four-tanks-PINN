import numpy as np
import copy


class StructTestContainer:
    """A container for the neural network structural test's data"""

    def __init__(self):
        # Results' structure: {'Layers = l1':
        #                          {'Neurons = n1':
        #                               {'train_u_loss': list,
        #                                'train_f_loss': list,
        #                                'train_total_loss': list,
        #                                'val_loss': list},
        #                           'Neurons = n2': {...}},
        #                      'Layers = l2': {...}, ...}
        self.results = dict()

        self.random_seed = None
        self.train_T = None
        self.np_train_u_X = None
        self.np_train_u_Y = None
        self.np_train_f_X = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None

    def check_key(self, layers, neurons):
        """
        Makes sure that the given keys of layers and neurons are registered in the results dictionary. If they are not
        registered yet, this function registers and links them to their respective dictionaries.

        :param layers: Key of layers ('Layers = l')
        :type layers: str
        :param neurons: Key of neurons ('Neurons = n')
        :type neurons: str
        """

        if layers not in self.results.keys():
            self.results[layers] = dict()
            self.results[layers][neurons] = dict()
        elif neurons not in self.results[layers].keys():
            self.results[layers][neurons] = dict()

    def get_final_val_losses(self, layers_group, neurons_group):
        """
        Returns a matrix with the final validation loss for each combination of layers and neurons. Layers number
        changes through the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Matrix with the final validation losses
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['val_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_total_losses(self, layers_group, neurons_group):
        """
        Returns a matrix with the final MSE for each combination of layers and neurons. Layers number changes through
        the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Final MSEs
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_total_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_u_losses(self, layers_group, neurons_group):
        """
        Returns a matrix with the final MSEu for each combination of layers and neurons. Layers number changes through
        the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Final MSEus
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_u_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_f_losses(self, layers_group, neurons_group):
        """
        Returns a matrix with the final MSEf for each combination of layers and neurons. Layers number changes through
        the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Final MSEfs
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_f_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_val_loss(self, layers, neurons):
        """
        Returns all the validation losses linked to the given combination of layers and neurons.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :returns: Validation losses
        :rtype: list
        """

        return self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['val_loss']

    def set_val_loss(self, layers, neurons, val_loss):
        """
        Sets the validation losses for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param val_loss: Validation losses
        :type val_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['val_loss'] = val_loss

    def set_train_total_loss(self, layers, neurons, train_total_loss):
        """
        Sets the train total MSEs for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_total_loss: Train total MSEs
        :type train_total_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, layers, neurons, train_u_loss):
        """
        Sets the train MSEus for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_u_loss: Train MSEus
        :type train_u_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, layers, neurons, train_f_loss):
        """
        Sets the train MSEfs for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_f_loss: Train MSEfs
        :type train_f_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_f_loss'] = train_f_loss


class TTestContainer:
    """A container for the neural network T test's data"""

    def __init__(self):
        # Train data's structure: {T1:
        #                              {'np_train_u_X': numpy.ndarray,
        #                               'np_train_u_Y': numpy.ndarray,
        #                               'np_train_f_X': numpy.ndarray},
        #                          T2: {...}, ...}
        self.train_data = dict()

        # Results' structure: {T1:
        #                          {'nn': numpy.ndarray,
        #                           'title': 'T = t s.',
        #                           'train_u_loss': list,
        #                           'train_f_loss': list,
        #                           'train_total_loss': list,
        #                           'val_loss': list},
        #                      T2: {...}, ...}
        self.results = dict()

        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None
        self.np_test_t = None
        self.np_test_X = None
        self.np_test_ic = None
        self.test_T = None
        self.np_test_Y = None

    def check_key(self, key, dictionary):
        """
        Makes sure that the given key is registered in the dictionary. If it is not, this function registers and links a
        new dictionary to it.

        :param key: Period T
        :type key: float
        :param dictionary: 'train_data' or 'results' dictionary
        :type dictionary: dict
        """

        if key not in dictionary.keys():
            dictionary[key] = dict()

    def get_train_u_X(self, train_T):
        """
        Returns the MSEu inputs linked to the given T.

        :param train_T: Period T
        :type train_T: float
        :returns: MSEu inputs
        :rtype: numpy.ndarray
        """

        return self.train_data[train_T]['np_train_u_X']

    def get_train_u_Y(self, train_T):
        """
        Returns the MSEu labels linked to the given T.

        :param train_T: Period T
        :type train_T: float
        :returns: MSEu labels
        :rtype: numpy.ndarray
        """

        return self.train_data[train_T]['np_train_u_Y']

    def get_train_f_X(self, train_T):
        """
        Returns the MSEf inputs linked to the given T.

        :param train_T: Period T
        :type train_T: float
        :returns: MSEf inputs
        :rtype: numpy.ndarray
        """

        return self.train_data[train_T]['np_train_f_X']

    def get_val_loss(self, train_T):
        """
        Returns all the validation losses linked to the given T.

        :param train_T: Period T
        :type train_T: float
        :returns: Validation losses
        :rtype: list
        """

        return self.results[train_T]['val_loss']

    def get_final_val_losses(self, train_Ts):
        """
        Returns the final validation loss for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Final validation losses
        :rtype: list
        """

        return np.array([self.results[T]['val_loss'][-1] for T in train_Ts])

    def get_final_train_total_losses(self, train_Ts):
        """
        Returns the final MSE for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Final MSEs
        :rtype: list
        """

        return np.array([self.results[T]['train_total_loss'][-1] for T in train_Ts])

    def get_final_train_u_losses(self, train_Ts):
        """
        Returns the final MSEu for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Final MSEus
        :rtype: list
        """

        return np.array([self.results[T]['train_u_loss'][-1] for T in train_Ts])

    def get_final_train_f_losses(self, train_Ts):
        """
        Returns the final MSEf for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Final MSEfs
        :rtype: list
        """

        return np.array([self.results[T]['train_f_loss'][-1] for T in train_Ts])

    def get_nn(self, train_T):
        """
        Returns the neural network's test output linked to the given T.

        :param train_T: Period T
        :type train_T: float
        :returns: Neural network's test output
        :rtype: numpy.ndarray
        """

        return self.results[train_T]['nn']

    def get_nns(self, train_Ts):
        """
        Returns the neural network's test output for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Neural network's test outputs
        :rtype: list
        """

        return [self.results[T]['nn'] for T in train_Ts]

    def get_titles(self, train_Ts):
        """
        Returns the plot title for each T.

        :param train_Ts: Periods T
        :type train_Ts: list or tuple
        :returns: Plot titles
        :rtype: list
        """

        return [self.results[T]['title'] for T in train_Ts]

    def get_results_dict(self):
        """
        Returns results dictionary with the numpy.ndarrays converted into lists.

        :returns: Results dictionary with the numpy.ndarrays converted into lists
        :rtype: dict
        """

        results = copy.deepcopy(self.results)
        for T in self.results.keys():
            results[T]['nn'] = results[T]['nn'].tolist()
        results['test_y'] = self.np_test_Y.tolist()
        results['test_t'] = self.np_test_t.tolist()

        return results

    def load_results(self, dictionary):
        """
        Loads the input dictionary's data into the class' results variable.

        :param dictionary: Loaded results
        :type dictionary: dict
        """

        self.np_test_t = np.array(dictionary['test_t'])
        self.np_test_Y = np.array(dictionary['test_y'])

        self.results = dictionary
        del self.results['test_t']
        del self.results['test_y']

        keys = list(self.results.keys())
        for key in keys:
            self.results[key]['nn'] = np.array(self.results[key]['nn'])
            self.results[float(key)] = self.results.pop(key)

    def set_train_u_X(self, train_T, np_train_u_X):
        """
        Sets the MSEu inputs for to the given T.

        :param train_T: Period T
        :type train_T: float
        :param np_train_u_X: MSEu inputs
        :type np_train_u_X: numpy.ndarray
        """

        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, train_T, np_train_u_Y):
        """
        Sets the MSEu labels for to the given T.

        :param train_T: Period T
        :type train_T: float
        :param np_train_u_Y: MSEu labels
        :type np_train_u_Y: numpy.ndarray
        """

        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, train_T, np_train_f_X):
        """
        Sets the MSEf inputs for to the given T.

        :param train_T: Period T
        :type train_T: float
        :param np_train_f_X: MSEf inputs
        :type np_train_f_X: numpy.ndarray
        """

        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_f_X'] = np_train_f_X

    def set_val_loss(self, train_T, val_loss):
        """
        Sets the validation losses for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param val_loss: Validation losses
        :type val_loss: list
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['val_loss'] = val_loss

    def set_train_total_loss(self, train_T, train_total_loss):
        """
        Sets the total MSEs for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param train_total_loss: Total MSEs
        :type train_total_loss: list
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, train_T, train_u_loss):
        """
        Sets the MSEus for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param train_u_loss: MSEus
        :type train_u_loss: list
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, train_T, train_f_loss):
        """
        Sets the MSEfs for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param train_f_loss: MSEfs
        :type train_f_loss: list
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['train_f_loss'] = train_f_loss

    def set_nn(self, train_T, np_nn):
        """
        Sets the neural network's test output for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param np_nn: Neural network's test output
        :type np_nn: numpy.ndarray
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['nn'] = np_nn

    def set_title(self, train_T, title):
        """
        Sets the plot title for to the given T in the results dictionary.

        :param train_T: Period T
        :type train_T: float
        :param title: Plot title
        :type title: str
        """

        self.check_key(train_T, self.results)
        self.results[train_T]['title'] = title


class NfNuTestContainer:
    """A container for the Nf/Nu test's data"""

    def __init__(self):
        # TODO: Set str or int as the only dictionaries' key type, for now they're both in use

        # Train datas' structure: {nf1:
        #                              {nu1:
        #                                   {'np_train_u_X': numpy.ndarray,
        #                                    'np_train_u_Y': numpy.ndarray,
        #                                    'np_train_f_X': numpy.ndarray},
        #                               nu2: {...},...},
        #                          nf2: {...}, ...}
        self.train_data = dict()

        # Results' structure: {'Nf = nf1':
        #                          {'Nu = nu1':
        #                               {'train_u_loss': list,
        #                                'train_f_loss': list,
        #                                'train_total_loss': list,
        #                                'val_loss': list},
        #                           'Nu = nu2': {...},...},
        #                      'Nf = nf2': {...}, ...}
        self.results = dict()

        self.random_seed = None
        self.train_T = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None

    def check_key(self, nf, nu, dictionary):
        """
        Makes sure that the given combination of Nf and Nu is registered in the dictionary. If it is not, this function
        registers and links a new dictionary to it.

        :param nf: Nf
        :type nf: int or str
        :param nu: Nu
        :type nu: int or str
        :param dictionary: 'train_data' or 'results' dictionary
        :type dictionary: dict
        """

        if nf not in dictionary.keys():
            dictionary[nf] = dict()
            dictionary[nf][nu] = dict()
        elif nu not in dictionary[nf].keys():
            dictionary[nf][nu] = dict()

    def get_train_u_X(self, nf, nu):
        """
        Returns the MSEu inputs linked to the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :returns: MSEu inputs
        :rtype: numpy.ndarray
        """

        return self.train_data[nf][nu]['np_train_u_X']

    def get_train_u_Y(self, nf, nu):
        """
        Returns the MSEu labels linked to the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :returns: MSEu labels
        :rtype: numpy.ndarray
        """

        return self.train_data[nf][nu]['np_train_u_Y']

    def get_train_f_X(self, nf, nu):
        """
        Returns the MSEf inputs linked to the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :returns: MSEf inputs
        :rtype: numpy.ndarray
        """

        return self.train_data[nf][nu]['np_train_f_X']

    def get_final_val_losses(self, nfs, nus):
        """
        Returns a matrix (numpy.ndarray) with the final validation loss for each combination of Nf and Nu. Nf changes
        through the rows, while Nu changes through the columns.

        :param nfs: Nfs of interest
        :type nfs: list or tuple
        :param nus: Nus of interest
        :type nus: list or tuple
        :returns: Matrix with the final validation losses
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['val_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_total_losses(self, nfs, nus):
        """
        Returns a matrix (numpy.ndarray) with the final MSE for each combination of Nf and Nu. Nf changes through the
        rows, while Nu changes through the columns.

        :param nfs: Nfs of interest
        :type nfs: list or tuple
        :param nus: Nus of interest
        :type nus: list or tuple
        :returns: Matrix with the final MSEs
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_total_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_u_losses(self, nfs, nus):
        """
        Returns a matrix (numpy.ndarray) with the final MSEu for each combination of Nf and Nu. Nf changes through the
        rows, while Nu changes through the columns.

        :param nfs: Nfs of interest
        :type nfs: list or tuple
        :param nus: Nus of interest
        :type nus: list or tuple
        :returns: Matrix with the final MSEus
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_u_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_f_losses(self, nfs, nus):
        """
        Returns a matrix (numpy.ndarray) with the final MSEf for each combination of Nf and Nu. Nf changes through the
        rows, while Nu changes through the columns.

        :param nfs: Nfs of interest
        :type nfs: list or tuple
        :param nus: Nus of interest
        :type nus: list or tuple
        :returns: Matrix with the final MSEfs
        :rtype: numpy.ndarray
        """

        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_f_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_val_loss(self, nf, nu):
        """
        Returns all the validation losses linked to the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :returns: Validation losses
        :rtype: list
        """

        return self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['val_loss']

    def set_train_u_X(self, nf, nu, np_train_u_X):
        """
        Sets the MSEu inputs for the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param np_train_u_X: MSEu inputs
        :type np_train_u_X: numpy.ndarray
        """

        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, nf, nu, np_train_u_Y):
        """
        Sets the MSEu labels for the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param np_train_u_Y: MSEu inputs
        :type np_train_u_Y: numpy.ndarray
        """

        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, nf, nu, np_train_f_X):
        """
        Sets the MSEf inputs for the given combination of Nf and Nu.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param np_train_f_X: MSEf inputs
        :type np_train_f_X: numpy.ndarray
        """

        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_f_X'] = np_train_f_X

    def set_val_loss(self, nf, nu, val_loss):
        """
        Sets the validation losses for the given combination of Nf and Nu in the results dictionary.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param val_loss: Validation losses
        :type val_loss: list
        """

        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['val_loss'] = val_loss

    def set_train_total_loss(self, nf, nu, train_total_loss):
        """
        Sets the total MSE for the given combination of Nf and Nu in the results dictionary.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param train_total_loss: Total MSE
        :type train_total_loss: list
        """

        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, nf, nu, train_u_loss):
        """
        Sets the MSEu for the given combination of Nf and Nu in the results dictionary.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param train_u_loss: MSEu
        :type train_u_loss: list
        """

        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, nf, nu, train_f_loss):
        """
        Sets the MSEf for the given combination of Nf and Nu in the results dictionary.

        :param nf: Nf
        :type nf: int
        :param nu: Nu
        :type nu: int
        :param train_f_loss: MSEf
        :type train_f_loss: list
        """

        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_f_loss'] = train_f_loss


class ExhaustionTestContainer:
    """A container for the exhaustion test's data"""

    def __init__(self):
        self.np_train_u_X = None
        self.np_train_u_Y = None
        self.np_train_f_X = None
        self.train_T = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None
        self.np_test_t = None
        self.np_test_X = None
        self.np_test_ic = None
        self.test_T = None
        self.np_test_Y = None
        self.np_test_NN = None
        self.train_total_loss = None
        self.train_u_loss = None
        self.train_f_loss = None
        self.val_loss = None

    def get_results_dict(self):
        """
        Returns all results in a dictionary.

        {'val_loss': val_loss,
         'train_total_loss': train_total_loss,
         'train_u_loss': train_u_loss,
         'train_f_loss': train_f_loss,
         'np_test_t': test_t,
         'np_test_U': test_U,
         'np_test_Y': test_Y,
         'np_test_NN': test_NN}.

         OBS: Every numpy.ndarray object is converted into list in the process.

        :returns: Results dictionary
        :rtype: dict
        """

        return {'val_loss': self.val_loss,
                'train_total_loss': self.train_total_loss,
                'train_u_loss': self.train_u_loss,
                'train_f_loss': self.train_f_loss,
                'np_test_t': self.np_test_t.tolist(),
                'np_test_U': self.get_np_test_U().tolist(),
                'np_test_Y': self.np_test_Y.tolist(),
                'np_test_NN': self.np_test_NN.tolist()}

    def get_np_test_U(self):
        """
        Returns the test's control inputs.

        :returns: Test's control inputs
        :rtype: numpy.ndarray
        """

        t_index = 0
        while not np.array_equal(self.np_test_X[:, t_index].flatten(), self.np_test_t.flatten()):
            t_index = t_index + 1
        return np.delete(self.np_test_X, t_index, axis=1)

    def load_results(self, dictionary):
        """
        Loads the input dictionary's data into the class' variables.

        :param dictionary: Loaded results
        :type dictionary: dict
        """

        self.val_loss = dictionary['val_loss']
        self.train_total_loss = dictionary['train_total_loss']
        self.train_u_loss = dictionary['train_u_loss']
        self.train_f_loss = dictionary['train_f_loss']

        self.np_test_t = np.array(dictionary['np_test_t'])
        test_t_matrix = np.transpose(np.array([dictionary['np_test_t']]))
        self.np_test_X = np.append(test_t_matrix, dictionary['np_test_U'], axis=1)
        self.np_test_Y = np.array(dictionary['np_test_Y'])
        self.np_test_NN = np.array(dictionary['np_test_NN'])
