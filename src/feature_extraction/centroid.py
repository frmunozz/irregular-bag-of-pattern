from sklearn.base import TransformerMixin, BaseEstimator
from scipy import sparse
import numpy as np


class CentroidClass(TransformerMixin, BaseEstimator):

    def __init__(self, classes=None):
        if classes is None:
            raise ValueError("need to give the classes labels")
        self.classes = classes
        self.n_classes = len(classes)

    def combine_vectors(self, X, y):
        n, bop_size = X.shape
        class_X = sparse.csr_matrix((self.n_classes, bop_size))
        class_counter = np.zeros(self.n_classes)
        for i in range(n):
            lbl = y[i]
            k = np.where(lbl == self.classes)[0]
            class_X[k] += X[i]
            class_counter[k] += 1

        for i in range(self.n_classes):
            class_X[i] /= class_counter[i]
        return class_X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("need labels to generate class based")
        return self.combine_vectors(X, y)
