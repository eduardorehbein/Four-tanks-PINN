import json
import numpy as np


class JsonDAO:
    def save(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def load(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


class TrainDataGenerator:
    def __init__(self, np_lowest_u, np_highest_u, np_lowest_y, np_highest_y):
        self.np_lowest_u = np_lowest_u
        self.np_highest_u = np_highest_u

        self.np_lowest_y = np_lowest_y
        self.np_highest_y = np_highest_y

    def get_data(self, scenarios, collocation_points, T, random_seed=30):
        # Set random seed
        np.random.seed(random_seed)

        # Controls and initial conditions
        np_us = np.random.uniform(low=self.np_lowest_u, high=self.np_highest_u, size=(scenarios, self.np_lowest_u.size))
        np_ics = np.random.uniform(low=self.np_lowest_y, high=self.np_highest_y, size=(scenarios, self.np_lowest_y.size))

        # Train u data
        np_train_u_X = np.concatenate([np.zeros((scenarios, 1)), np_us, np_ics], axis=1)
        np_train_u_Y = np_ics

        # Train f data
        np_t = np.reshape(np.linspace(0, T, collocation_points), (collocation_points, 1))

        np_train_f_X = np.concatenate([np_t,
                                       np.tile(np_us[0, :], (collocation_points, 1)),
                                       np.tile(np_ics[0, :], (collocation_points, 1))], axis=1)

        for i in range(1, scenarios):
            np_u = np.tile(np_us[i, :], (collocation_points, 1))
            np_ic = np.tile(np_ics[i, :], (collocation_points, 1))

            np_X = np.concatenate([np_t, np_u, np_ic], axis=1)
            np_train_f_X = np.append(np_train_f_X, np_X, axis=0)

        return np_train_u_X, np_train_u_Y, np_train_f_X

