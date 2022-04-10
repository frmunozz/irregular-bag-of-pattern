from sklearn.neighbors import KNeighborsClassifier as skl_knn
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
# from dtaidistance.dtw_ndim import distance_fast


class KNeighborsClassifier(skl_knn):

    def __init__(self, n_neighbors=1, classes=None, useClasses=False, **kwargs):
        self.classes = classes
        self.useClasses = useClasses
        super(KNeighborsClassifier, self).__init__(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        if self.useClasses:
            return super(KNeighborsClassifier, self).fit(X, self.classes.ravel())
        else:
            return super(KNeighborsClassifier, self).fit(X, y)


# def distance_matrix_ddtw(X):
#     n_observations, n_features, n_variables = X.shape
#     X_out = np.zeros((n_observations, n_observations), dtype=np.double)
#     X = X.astype(np.double)
#     for i in range(n_observations):
#         for j in range(i, n_observations):
#             d = distance_fast(X[i], X[j])
#             X_out[i,j] = d
#             X_out[j,i] = d
#     return X_out
#
#
# def flattened_ddtw(x1, x2, shape=None):
#     if shape is None:
#         raise ValueError("shape needed")
#     # b bands/variables
#     assert len(x1) % shape[1] == 0
#     assert len(x2) % shape[1] == 0
#
#     arr1 = x1.reshape(shape).astype(np.double)
#     arr2 = x2.reshape(shape).astype(np.double)
#
#     return distance_fast(arr1, arr2)


# class DependentDTWMatrix(TransformerMixin, BaseEstimator):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X
#
#     def fit_transform(self, X, y=None, **fit_params):
#         return distance_matrix_ddtw(X)


class Flatten3Dto2D(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_observations, n_features, n_variables = X.shape
        return X.reshape(n_observations, n_features * n_variables)

    def fit_transform(self, X, y=None, **fit_params):
        n_observations, n_features, n_variables = X.shape
        return X.reshape(n_observations, n_features * n_variables)



