import numpy as np


class Normalizer:
    """Neural network's data normalizer."""

    def __init__(self, dictionary=None, analysis_axis=0):
        """
        Defines the analysis axis. If it is 0, the class' functions will work with row vectors, if it is 1, column
        vectors. It also makes possible the normalizer's parameters initialization through a preloaded dictionary.

        :param dictionary: Preloaded dictionary with the normalizer's parameters
            (default is None)
        :type dictionary: dict
        :param analysis_axis: Analysis axis
        :type analysis_axis: int
        """
        if dictionary is not None:
            self.__dict__.update(dictionary)
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
        else:
            self.analysis_axis = analysis_axis
            self.mean = None
            self.std = None

    def parametrize(self, data):
        """
        Defines the class' parameters using the given set of data.

        :param data: Data
        :type data: numpy.ndarray or tensorflow.Tensor
        """

        self.mean = np.array([data.mean(axis=self.analysis_axis)])
        self.std = np.array([data.std(axis=self.analysis_axis)])

        if self.analysis_axis == 1:
            self.mean = np.transpose(self.mean)
            self.std = np.transpose(self.std)

    def normalize(self, data):
        """
        Returns the input data normalized based on the class' parameters. If these parameters are undefined, the
        function uses the input data to define them before the normalization.

        :param data: Data
        :type data: numpy.ndarray or tensorflow.Tensor
        :returns: Normalized data
        :rtype: numpy.ndarray or tensorflow.Tensor
        """

        if self.mean is None or self.std is None:
            self.parametrize(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """
        Returns the input data denormalized based on the class' parameters. If these parameters are undefined, the
        function throws an exception.

        :param data: Data
        :type data: numpy.ndarray or tensorflow.Tensor
        :returns: Denormalized data
        :rtype: numpy.ndarray or tensorflow.Tensor
        """

        if self.mean is None or self.std is None:
            raise Exception('Undefined params for denormalization, the normalizer need to be parametrized.')
        else:
            return data * self.std + self.mean
