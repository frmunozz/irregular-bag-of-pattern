from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
import numpy as np


class VSM(TfidfTransformer):
    def __init__(self, class_based=False, classes=None, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, return_dense=False):
        super(VSM, self).__init__(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                                  sublinear_tf=sublinear_tf)
        self.class_based = class_based
        self.classes = classes
        self.n_classes = len(classes) if classes is not None else 0
        self.return_dense = return_dense

    def combine_vectors(self, X, y):
        n, bop_size = X.shape
        class_X = sparse.lil_matrix((self.n_classes, bop_size))
        for i in range(n):
            lbl = y[i]
            k = np.where(lbl == self.classes)[0]
            class_X[k] += X[i]
        return class_X

    def get_scheme_notation(self):
        scheme = "l" if self.sublinear_tf else "n"
        scheme += "t" if self.use_idf else "n"
        scheme += "c" if self.norm == "l2" else "n"
        return scheme

    def fit(self, X, y=None):
        if self.class_based:
            if y is None:
                raise ValueError("need labels to generate class based")
            class_X = self.combine_vectors(X, y)
            return super(VSM, self).fit(class_X, y=self.classes)

        return super(VSM, self).fit(X, y=y)

    def fit_transform(self, X, y=None, **fit_params):
        if self.class_based:
            if y is None:
                raise ValueError("need labels to generate class based")
            X = self.combine_vectors(X, y)
        r = super(VSM, self).fit_transform(X, y=y, **fit_params)
        if self.return_dense:
            try:
                r = r.todense()
            except:
                print("cannot generate dense")
        return r

    def transform(self, X):
        r = super(VSM, self).transform(X)
        if self.return_dense:
            try:
                r = r.todense()
            except:
                print("cannot generate dense")
        return r