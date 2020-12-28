from sklearn.feature_selection import SelectKBest as skl_SelectKBest
from sklearn.feature_selection import f_classif

from ..utils import get_vocabulary_size


class SelectKTop(skl_SelectKBest):

    def __init__(self, sc=None, score_func=f_classif):
        if sc is  None:
            raise ValueError("need to define an spatial complexity")
        self.sc = sc
        super(SelectKTop, self).__init__(score_func=score_func, k=sc-1)

    def fit(self, X, y):
        _, bop_size = X.shape
        self.k = min(self.sc - 1, bop_size - 1)
        return super(SelectKTop, self).fit(X, y)
