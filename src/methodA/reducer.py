from ..utils import AbstractCore
import copy
from .representation import BOPSparseRepresentation
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from abc import abstractmethod


class BaseReducer(AbstractCore):

    @classmethod
    def module_name(cls):
        return "Reducer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reducer = None

    @abstractmethod
    def fit(self, vectors: BOPSparseRepresentation, labels: np.ndarray = None):
        pass

    @abstractmethod
    def transform(self, vectors: BOPSparseRepresentation) -> BOPSparseRepresentation:
        pass


class PCAReducer(PCA):

    def __init__(self, **kwargs):
        super().__init__(n_components=kwargs.get("n_components", None))
        self.variance_ratio = kwargs.get("variance_ratio", None)
        if self.variance_ratio is None and self.n_components is None:
            raise ValueError("must define a number of components or a variance ratio threshold")

    def fit(self, vectors: BOPSparseRepresentation, labels: np.ndarray = None):
        super(PCAReducer, self).fit(vectors.to_array())

    def transform(self, vectors: BOPSparseRepresentation) -> BOPSparseRepresentation:
        out_vectors = super(PCAReducer, self).transform(vectors.to_array())
        if self.variance_ratio:
            tmp = np.cumsum(self.explained_variance_ratio_)
            idx = np.argmax(tmp > self.variance_ratio)
            out_vectors = out_vectors[:, :idx]
        return BOPSparseRepresentation(out_vectors)

    def fit_transform(self, vectors: BOPSparseRepresentation, y=None):
        out_vectors = super(PCAReducer, self).fit_transform(vectors.to_array(), y=y)
        if self.variance_ratio:
            tmp = np.cumsum(self.explained_variance_ratio_)
            idx = np.argmax(tmp > self.variance_ratio)
            out_vectors = out_vectors[:, :idx]
        return BOPSparseRepresentation(out_vectors)

    def total_variance_ratio(self):
        return np.sum(self.explained_variance_ratio_)


# class LSAReducer(TruncatedSVD):
#
#     def __init__(self, n_components):
#         super(LSAReducer, self).__init__(n_components=n_components)
#
#     def fit(self, vectors, y=None):
#         super(LSAReducer, self).fit(vectors.to_array())
#
#     def fit_transform(self, X, y=None):
#         out_vectors = super(LSAReducer, self).fit_transform(X.to_array(), y=y)
#         out_obj = BOPSparseRepresentation()
#         out_obj.store_repr(out_vectors)


_ANOVA_KWARGS = {

}


class ANOVAReducer(AbstractCore):
    def get_valid_kwargs(self) -> dict:
        return copy.deepcopy(_ANOVA_KWARGS)

    @classmethod
    def module_name(cls):
        return "ANOVAReducer"

