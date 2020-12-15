from scipy import sparse
import numpy as np
from sklearn.preprocessing import StandardScaler


class BOPSparseRepresentation(object):

    def __init__(self, _format="lil"):
        self._format = _format
        self.vector = None
        self.scaler = None
        self.scale = False

    def store_repr(self, vector):
        if isinstance(vector, (list, np.ndarray)):
            if self.format == "lil":
                vector = sparse.lil_matrix(vector)
            else:
                vector = sparse.csr_matrix(vector)
        self.vector = vector

    def hstack_repr(self, other):
        assert self.format == other.format
        assert self.vector.shape[0] == other.vector.shape[0]
        self.vector = sparse.hstack((self.vector, other.vector),
                                    format=self.format)

    def vstack_repr(self, other):
        assert self.format == other.format
        assert self.vector.shape[1] == other.vector.shape[1]

        self.vector = sparse.vstack((self.vector, other.vector),
                                    format=self.format)

    @property
    def format(self):
        return self._format

    def to_dense(self):
        return self.vector.todense()

    def to_array(self):
        arr = self.vector.toarray()
        if self.scale:
            arr = self.scaler.transform(arr)
        return arr

    @property
    def size(self):
        if self.vector is None:
            return 0
        return self.vector.size

    def copy_from(self, other):
        self._format = other._format
        self.vector = other.vector.copy()

    def set_scaler(self, scaler, with_fit=True):
        self.scaler = scaler
        self.scale = True
        if with_fit:
            self.scaler.fit(self.vector.toarray())
