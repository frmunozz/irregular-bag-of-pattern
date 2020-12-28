from sklearn.feature_selection import SelectKBest as skl_SelectKBest
from sklearn.feature_selection import f_classif

from ..utils import get_vocabulary_size


class SelectKBest(skl_SelectKBest):

    def __init__(self, spatial_complexity, score_func=f_classif):
        super(SelectKBest, self).__init__(score_func=score_func)
        self.k = spatial_complexity - 1
        self._sc = spatial_complexity

    def fit(self, X, y):
        _, bop_size = X.shape
        self.k = min(self._sc - 1, bop_size - 1)
        return super(SelectKBest, self).fit(X, y)
