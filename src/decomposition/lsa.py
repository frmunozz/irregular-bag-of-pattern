from sklearn.decomposition import TruncatedSVD


class LSA(TruncatedSVD):
    def __init__(self, sc=None, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        if sc is None:
            raise ValueError("need to define an spatial complexity")
        super(LSA, self).__init__(algorithm=algorithm, n_iter=n_iter,
                                  random_state=random_state, tol=tol)
        self.n_components = sc - 1
        self.sc = sc
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        _, bop_size = X.shape
        self.n_components = min(self.sc - 1, bop_size - 1)
        # print("update n components in fit to:", self.n_components)
        return super(LSA, self).fit(X, y=y)

    def fit_transform(self, X, y=None):
        _, bop_size = X.shape
        self.n_components = min(self.sc - 1, bop_size - 1)
        # print("update n components in fit_transform to:", self.n_components)
        return super(LSA, self).fit_transform(X, y=y)
