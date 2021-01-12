import json
import numpy as np


class JsonDAO:
    """JSON data access object."""

    def save(self, file_path, data):
        """
        It saves the given data dictionary in a JSON file.

        :param file_path: File path
        :type file_path: str
        :param data: Data dictionary
        :type data: dict
        """

        with open(file_path, 'w') as file:
            json.dump(data, file)

    def load(self, file_path):
        """
        It loads the specified JSON file and returns a dictionary containing its data.

        :param file_path: File path
        :type file_path: str
        :returns: Data dictionary
        :rtype: dict
        """

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


class TrainDataGenerator:
    """It generates training data based on some parameters."""

    def __init__(self, np_lowest_u, np_highest_u, np_lowest_y, np_highest_y):
        """
        It sets constraints for the generation of control inputs and initial conditions.

        :param np_lowest_u: Lowest control inputs' values
        :type np_lowest_u: numpy.ndarray
        :param np_highest_u: Highest control inputs' values
        :type np_highest_u: numpy.ndarray
        :param np_lowest_y: Lowest initial conditions' values
        :type np_lowest_y: numpy.ndarray
        :param np_highest_y: Highest initial conditions' values
        :type np_highest_y: numpy.ndarray
        """

        self.np_lowest_u = np_lowest_u
        self.np_highest_u = np_highest_u

        self.np_lowest_y = np_lowest_y
        self.np_highest_y = np_highest_y

    def get_data(self, scenarios, collocation_points, T, random_seed=30):
        """
        It returns the whole set of train data. The set is generated using the given parameters and the preset constraints.

        :param scenarios: Number of different (u, y0) pairs to be generated
        :type scenarios: int
        :param collocation_points: Number of points to generate for the MSEf data between the instants 0 and T,
            including them
        :type collocation_points: int
        :param T: Maximum time value for the MSEf data generation
        :type T: float
        :param random_seed: Random seed
            (default is 30)
        :type random_seed: int
        :returns: MSEu inputs, MSEu labels, MSEf inputs
        :rtype: tuple
        """

        # Random seed
        np.random.seed(random_seed)

        # Controls and initial conditions
        np_us = np.random.uniform(low=self.np_lowest_u, high=self.np_highest_u, size=(scenarios, self.np_lowest_u.size))
        np_ics = np.random.uniform(low=self.np_lowest_y, high=self.np_highest_y, size=(scenarios, self.np_lowest_y.size))

        # MSEu data
        np_train_u_X = np.concatenate([np.zeros((scenarios, 1)), np_us, np_ics], axis=1)
        np_train_u_Y = np_ics

        # MSEf data
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

