import numpy as np


class UTS:
    def __init__(self, y, metadata=None):
        """
        Regular Univariate Time Series data type
        :param y: 1D data array
        """
        if isinstance(y, list):
            y = np.array(y)
        if metadata is None:
            metadata = {}
        self.y = y
        self.metadata = metadata

    def mean(self):
        return np.mean(self.y)

    def get_sequence(self, i, j):
        return UTS(self.y[i:j])

    def size(self):
        return self.y.size


class IrregularUTS(UTS):
    def __init__(self, t, y, metadata=None):
        """
        Irregular Univariate Time Series data type
        :param t: 1D time array
        :param y: 1D data array
        """
        super().__init__(y, metadata=metadata)
        if isinstance(t, list):
            t = np.array(t)
        self.t = t
        if len(self.t) != len(self.y):
            raise ValueError(
                "imput vectors dimension doesnt match: ({}) for time vector, ({}) for value vector".format(len(self.t),
                                                                                                           len(self.y)))

    def bandwidth(self):
        return self.end_time() - self.t[0]

    def end_time(self):
        if self.size() > 0:
            return self.t[self.size() - 1]
        else:
            return 0

    def ini_time(self):
        if self.size() > 0:
            return self.t[0]
        else:
            return 0

    def get_time(self, idx):
        if idx >= self.size():
            return np.inf
        return self.t[idx]

    def get_sequence(self, i, j):
        return IrregularUTS(self.t[i:j], self.y[i:j])

    def data(self):
        return zip(self.t, self.y)


class EmptyIrregularUTS(IrregularUTS):
    def __init__(self):
        """
        Empty object of an irregular UTS
        """
        super().__init__(np.array([]), np.array([]))

    def size(self):
        return 0
