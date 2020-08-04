import numpy as np


class Normalizer:
    def __init__(self, dictionary=None, analysis_axis=0):
        if dictionary is not None:
            self.__dict__.update(dictionary)
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
        else:
            self.analysis_axis = analysis_axis
            self.mean = None
            self.std = None

    def parametrize(self, data):
        self.mean = np.array([data.mean(axis=self.analysis_axis)])
        self.std = np.array([data.std(axis=self.analysis_axis)])

        if self.analysis_axis == 1:
            self.mean = np.transpose(self.mean)
            self.std = np.transpose(self.std)

    def normalize(self, data):
        if self.mean is None or self.std is None:
            self.parametrize(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if self.mean is None or self.std is None:
            raise Exception('Undefined params for denormalization, the normalizer need to be parametrized.')
        else:
            return data * self.std + self.mean
