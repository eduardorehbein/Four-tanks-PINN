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
        Makes sure that the given keys of layers and neurons are registered in the results dictionary.

        :param layers: Key of layers ('Layers = a')
        :type layers: str
        :param neurons: Key of neurons ('Neurons = b')
        :type neurons: str
        """

        if layers not in self.results.keys():
            self.results[layers] = dict()
            self.results[layers][neurons] = dict()
        elif neurons not in self.results[layers].keys():
            self.results[layers][neurons] = dict()

    def get_final_val_losses(self, layers_group, neurons_group):
        """
        Returns a matrix list(list) with the final validation losses for each combination of layers and neurons. Layers
        number changes through the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Matrix with the final validation losses
        :rtype: list
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['val_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_total_losses(self, layers_group, neurons_group):
        """
        Returns a matrix list(list) with the final train total losses for each combination of layers and neurons. Layers
        number changes through the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Matrix with the final train total losses
        :rtype: list
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_total_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_u_losses(self, layers_group, neurons_group):
        """
        Returns a matrix list(list) with the final train u losses for each combination of layers and neurons. Layers
        number changes through the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Matrix with the final train u losses
        :rtype: list
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_u_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_f_losses(self, layers_group, neurons_group):
        """
        Returns a matrix list(list) with the final train f losses for each combination of layers and neurons. Layers
        number changes through the rows, while neurons number changes through the columns.

        :param layers_group: Layers numbers of interest
        :type layers_group: list or tuple
        :param neurons_group: Neurons numbers of interest
        :type neurons_group: list or tuple
        :returns: Matrix with the final train f losses
        :rtype: list
        """

        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_f_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_val_loss(self, layers, neurons):
        """
        Returns all the validation losses for the given combination of layers and neurons.

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
        Sets the train total losses for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_total_loss: Train total losses
        :type train_total_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, layers, neurons, train_u_loss):
        """
        Sets the train u losses for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_u_loss: Train u losses
        :type train_u_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, layers, neurons, train_f_loss):
        """
        Sets the train f losses for the given combination of layers and neurons in the results dictionary.

        :param layers: Number of layers
        :type layers: int
        :param neurons: Number of Neurons
        :type neurons: int
        :param train_f_loss: Train f losses
        :type train_f_loss: list
        """

        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_f_loss'] = train_f_loss


class TTestContainer:
    def __init__(self):
        self.train_data = dict()
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
        if key not in dictionary.keys():
            dictionary[key] = dict()

    def get_train_u_X(self, train_T):
        return self.train_data[train_T]['np_train_u_X']

    def get_train_u_Y(self, train_T):
        return self.train_data[train_T]['np_train_u_Y']

    def get_train_f_X(self, train_T):
        return self.train_data[train_T]['np_train_f_X']

    def get_val_loss(self, train_T):
        return self.results[train_T]['val_loss']

    def get_final_val_losses(self, train_Ts):
        return np.array([self.results[T]['val_loss'][-1] for T in train_Ts])

    def get_final_train_total_losses(self, train_Ts):
        return np.array([self.results[T]['train_total_loss'][-1] for T in train_Ts])

    def get_final_train_u_losses(self, train_Ts):
        return np.array([self.results[T]['train_u_loss'][-1] for T in train_Ts])

    def get_final_train_f_losses(self, train_Ts):
        return np.array([self.results[T]['train_f_loss'][-1] for T in train_Ts])

    def get_nn(self, train_T):
        return self.results[train_T]['nn']

    def get_nns(self, train_Ts):
        return [self.results[T]['nn'] for T in train_Ts]

    def get_titles(self, train_Ts):
        return [self.results[T]['title'] for T in train_Ts]

    def get_results_dict(self):
        results = copy.deepcopy(self.results)
        for T in self.results.keys():
            results[T]['nn'] = results[T]['nn'].tolist()
        results['test_y'] = self.np_test_Y.tolist()
        results['test_t'] = self.np_test_t.tolist()

        return results

    def load_results(self, dictionary):
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
        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, train_T, np_train_u_Y):
        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, train_T, np_train_f_X):
        self.check_key(train_T, self.train_data)
        self.train_data[train_T]['np_train_f_X'] = np_train_f_X

    def set_val_loss(self, train_T, val_loss):
        self.check_key(train_T, self.results)
        self.results[train_T]['val_loss'] = val_loss

    def set_train_total_loss(self, train_T, train_total_loss):
        self.check_key(train_T, self.results)
        self.results[train_T]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, train_T, train_u_loss):
        self.check_key(train_T, self.results)
        self.results[train_T]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, train_T, train_f_loss):
        self.check_key(train_T, self.results)
        self.results[train_T]['train_f_loss'] = train_f_loss

    def set_nn(self, train_T, np_nn):
        self.check_key(train_T, self.results)
        self.results[train_T]['nn'] = np_nn

    def set_title(self, train_T, title):
        self.check_key(train_T, self.results)
        self.results[train_T]['title'] = title


class NfNuTestContainer:
    def __init__(self):
        self.train_data = dict()
        self.results = dict()
        self.random_seed = None
        self.train_T = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None

    def check_key(self, nf, nu, dictionary):
        if nf not in dictionary.keys():
            dictionary[nf] = dict()
            dictionary[nf][nu] = dict()
        elif nu not in dictionary[nf].keys():
            dictionary[nf][nu] = dict()

    def get_train_u_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_X']

    def get_train_u_Y(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_Y']

    def get_train_f_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_f_X']

    def get_final_val_losses(self, nfs, nus):
        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['val_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_total_losses(self, nfs, nus):
        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_total_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_u_losses(self, nfs, nus):
        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_u_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_final_train_f_losses(self, nfs, nus):
        return np.array([[self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['train_f_loss'][-1] for nu in nus]
                         for nf in nfs])

    def get_val_loss(self, nf, nu):
        return self.results['Nf = ' + str(nf)]['Nu = ' + str(nu)]['val_loss']

    def set_train_u_X(self, nf, nu, np_train_u_X):
        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, nf, nu, np_train_u_Y):
        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, nf, nu, np_train_f_X):
        self.check_key(nf, nu, self.train_data)
        self.train_data[nf][nu]['np_train_f_X'] = np_train_f_X

    def set_val_loss(self, nf, nu, val_loss):
        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['val_loss'] = val_loss

    def set_train_total_loss(self, nf, nu, train_total_loss):
        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, nf, nu, train_u_loss):
        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, nf, nu, train_f_loss):
        nf_key = 'Nf = ' + str(nf)
        nu_key = 'Nu = ' + str(nu)
        self.check_key(nf_key, nu_key, self.results)
        self.results[nf_key][nu_key]['train_f_loss'] = train_f_loss


class ExhaustionTestContainer:
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
        return {'val_loss': self.val_loss,
                'train_total_loss': self.train_total_loss,
                'train_u_loss': self.train_u_loss,
                'train_f_loss': self.train_f_loss,
                'np_test_t': self.np_test_t.tolist(),
                'np_test_U': self.get_np_test_U().tolist(),
                'np_test_Y': self.np_test_Y.tolist(),
                'np_test_NN': self.np_test_NN.tolist()}

    def get_np_test_U(self):
        t_index = 0
        while not np.array_equal(self.np_test_X[:, t_index].flatten(), self.np_test_t.flatten()):
            t_index = t_index + 1
        return np.delete(self.np_test_X, t_index, axis=1)

    def load_results(self, dictionary):
        self.val_loss = dictionary['val_loss']
        self.train_total_loss = dictionary['train_total_loss']
        self.train_u_loss = dictionary['train_u_loss']
        self.train_f_loss = dictionary['train_f_loss']

        self.np_test_t = np.array(dictionary['np_test_t'])
        test_t_matrix = np.transpose(np.array([dictionary['np_test_t']]))
        self.np_test_X = np.append(test_t_matrix, dictionary['np_test_U'], axis=1)
        self.np_test_Y = np.array(dictionary['np_test_Y'])
        self.np_test_NN = np.array(dictionary['np_test_NN'])
