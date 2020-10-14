import numpy as np
import copy


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

    def check_key(self, key, directory):
        if key not in directory.keys():
            directory[key] = dict()

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
        self.train_T = None
        self.np_val_X = None
        self.np_val_ic = None
        self.val_T = None
        self.np_val_Y = None

    def check_key(self, nf, nu):
        if nf not in self.train_data.keys():
            self.train_data[nf] = dict()
            self.train_data[nf][nu] = dict()
        elif nu not in self.train_data[nf].keys():
            self.train_data[nf][nu] = dict()

    def get_train_u_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_X']

    def get_train_u_Y(self, nf, nu):
        return self.train_data[nf][nu]['np_train_u_Y']

    def get_train_f_X(self, nf, nu):
        return self.train_data[nf][nu]['np_train_f_X']

    def set_train_u_X(self, nf, nu, np_train_u_X):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_u_X'] = np_train_u_X

    def set_train_u_Y(self, nf, nu, np_train_u_Y):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_u_Y'] = np_train_u_Y

    def set_train_f_X(self, nf, nu, np_train_f_X):
        self.check_key(nf, nu)
        self.train_data[nf][nu]['np_train_f_X'] = np_train_f_X
