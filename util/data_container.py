import numpy as np
import copy


class StructTestContainer:
    def __init__(self):
        self.results = dict()
        self.train_T = None
        self.np_train_u_X = None
        self.np_train_u_Y = None
        self.np_train_f_X = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None

    def check_key(self, layers, neurons):
        if layers not in self.results.keys():
            self.results[layers] = dict()
            self.results[layers][neurons] = dict()
        elif neurons not in self.results[layers].keys():
            self.results[layers][neurons] = dict()

    def get_final_val_losses(self, layers_group, neurons_group):
        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['val_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_total_losses(self, layers_group, neurons_group):
        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_total_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_u_losses(self, layers_group, neurons_group):
        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_u_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def get_final_train_f_losses(self, layers_group, neurons_group):
        return np.array([[self.results['Layers = ' + str(layers)]['Neurons = ' + str(neurons)]['train_f_loss'][-1]
                          for neurons in neurons_group] for layers in layers_group])

    def set_val_loss(self, layers, neurons, val_loss):
        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['val_loss'] = val_loss

    def set_train_total_loss(self, layers, neurons, train_total_loss):
        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_total_loss'] = train_total_loss

    def set_train_u_loss(self, layers, neurons, train_u_loss):
        layers_key = 'Layers = ' + str(layers)
        neurons_key = 'Neurons = ' + str(neurons)
        self.check_key(layers_key, neurons_key)
        self.results[layers_key][neurons_key]['train_u_loss'] = train_u_loss

    def set_train_f_loss(self, layers, neurons, train_f_loss):
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
