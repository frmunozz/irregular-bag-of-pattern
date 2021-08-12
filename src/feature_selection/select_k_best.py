from sklearn.feature_selection import SelectKBest as skl_SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
from ..utils import get_vocabulary_size
from sklearn.utils.validation import check_array
from sklearn.utils import safe_mask
from warnings import warn


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


class GeneralSelectKTop(skl_SelectKBest):

    def __init__(self, k, score_func, allow_nd=True, n_variables=None):
        if n_variables is None:
            raise ValueError("need the n_variables")
        self.allow_nd = False  # to use X of 3 dims ##TODO: not using?
        self.n_variables = n_variables
        super(GeneralSelectKTop, self).__init__(k=k, score_func=score_func)

    def fit(self, X, y):
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'], allow_nd=self.allow_nd,
                                   multi_output=True)
        # if X.ndim != 3:
        #     raise ValueError("Data array dimensions invalid: %d != 3" % X.dim)

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self._check_params(X, y)
        score_func_ret = self.score_func(X, y, self.n_variables)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
            # self.k = len(np.where(self.pvalues_ < 0.5)[0])
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        self.scores_ = np.asarray(self.scores_)

        return self

    def _get_support_mask(self):
        # print("scores shape: ", self.scores_.shape)
        # print("nan in scores: ", )
        mask = super(GeneralSelectKTop, self)._get_support_mask()
        idxs = np.argsort(self.scores_, kind="mergesort")[-self.k:]
        # print(idxs[0], self.scores_[idxs[0]], self.pvalues_[idxs[0]], idxs[-1], self.scores_[idxs[-1]], self.pvalues_[idxs[-1]], np.max(self.pvalues_))
        # print("min selected score:", np.min(self.scores_[mask]), ", max selected score:", np.max(self.scores_[mask]))
        # print("mask shape:", mask.shape)
        mask_extended = np.zeros(mask.shape[0] * self.n_variables, dtype=bool)
        for i, value in enumerate(mask):
            for j in range(self.n_variables):
                mask_extended[i * self.n_variables + j] = value
        return mask_extended

    # def transform(self, X):
    #     """Reduce X to the selected features.
    #
    #     Parameters
    #     ----------
    #     X : array of shape [n_samples, n_features]
    #         The input samples.
    #
    #     Returns
    #     -------
    #     X_r : array of shape [n_samples, n_selected_features]
    #         The input samples with only the selected features.
    #     """
    #     tags = self._get_tags()
    #     X = check_array(X, dtype=None, accept_sparse='csr', allow_nd=self.allow_nd,
    #                     force_all_finite=not tags.get('allow_nan', True))
    #     mask = self.get_support()
    #     if not mask.any():
    #         warn("No features were selected: either the data is"
    #              " too noisy or the selection test too strict.",
    #              UserWarning)
    #         return np.empty(0).reshape((X.shape[0], 0))
    #     if len(mask) != X.shape[1]:
    #         raise ValueError("X has a different shape than during fitting.")
    #     return X[:, safe_mask(X, mask)]

    def transform(self, X):
        ret = super(GeneralSelectKTop, self).transform(X)
        if not isinstance(ret, np.ndarray):
            ret = ret.toarray()
        # print("FORMAT RESULTING FEATURE SELECTED:", type(ret))
        return ret

    def fit_transform(self, X, y=None, **fit_params):
        ret = super(GeneralSelectKTop, self).fit_transform(X, y=y, **fit_params)
        if not isinstance(ret, np.ndarray):
            ret = ret.toarray()
        # print("FORMAT RESULTING FEATURE SELECTED:", type(ret))
        return ret