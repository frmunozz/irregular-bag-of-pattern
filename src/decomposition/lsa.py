from sklearn.decomposition import TruncatedSVD


class LSA(TruncatedSVD):
    def __init__(self, spatial_complexity, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        super(LSA, self).__init__(algorithm=algorithm, n_iter=n_iter,
                                  random_state=random_state, tol=tol)
        self.n_components = spatial_complexity - 1
        self._sc = spatial_complexity

    def fit(self, X, y=None):
        _, bop_size = X.shape
        self.n_components = min(self._sc - 1, bop_size - 1)
        return super(LSA, self).fit(X, y=y)
