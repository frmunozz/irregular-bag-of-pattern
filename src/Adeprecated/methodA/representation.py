from scipy import sparse
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from .class_vectors import compute_class_centroids, compute_class_tf_idf


class BOPSparseRepresentation(object):

    def __init__(self, _format="lil", zeros_shape=None):
        self._format = _format
        self.vector = None
        self.scaler = None
        self.scale = False
        if zeros_shape is not None:
            if self.format == "lil":
                self.vector = sparse.lil_matrix(zeros_shape)
            else:
                self.vector = sparse.csr_matrix(zeros_shape)

        self.status = "count_vectorizer"

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
        self.vector = sparse.hstack((self.vector, other.vector.copy()),
                                    format=self.format)

    def vstack_repr(self, other):
        assert self.format == other.format
        assert self.vector.shape[1] == other.vector.shape[1]

        self.vector = sparse.vstack((self.vector, other.vector.copy()),
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

    def as_tf_idf(self, transformer=None):
        if self.status == "count_vectorizer":
            if transformer is None:
                transformer = TfidfTransformer()
            self.vector = transformer.fit_transform(self.vector)
            self.status = "tf_idf"
        elif self.status == "tf_idf":
            pass
        else:
            raise ValueError("cant transform vector from {} to TF_IDF".format(self.status.upper()))

    def as_class_centroid(self, labels):
        if self.status == "count_vectorizer":
            self.vector, classes = compute_class_centroids(self.vector, labels)
            self.status = "class_centroid"
        elif self.status == "class_centroid":
            pass
        else:
            raise ValueError("cant transform vector from {} to CLASS CENTROID".format(self.status.upper()))

    def as_class_tf_idf(self, labels):
        if self.status == "count_vectorizer":
            self.vector, classes = compute_class_tf_idf(self.vector, labels)
            self.status = "class_tf_idf"
        elif self.status == "class_tf_idf":
            pass
        else:
            raise ValueError("cant transform vector from {} to CLASS TF IDF".format(self.status.upper()))

    def sample_wise_norm(self):
        self.vector = normalize(self.vector)

    def count_failed(self) -> int:
        sums = np.sum(self.vector, axis=1)
        failed = len(np.where(sums == 0)[0])
        return failed
