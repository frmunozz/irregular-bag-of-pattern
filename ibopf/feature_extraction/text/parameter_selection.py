from sklearn.base import TransformerMixin, BaseEstimator


class ParameterSelector(TransformerMixin, BaseEstimator):
    def __init__(self, idx=None, data=None, win_arr=None, wl_arr=None):
        if idx is None or data is None:
            raise ValueError("need to set idx and data")
        self.idx = idx
        self.data = data
        self.win_arr = win_arr
        self.wl_arr = wl_arr

    def fit(self, X, y=None, **kwargs):
        # print("gettign the count words for win idx: %f.3 and wl idx: %d" % (
        #     self.win_arr[self.idx], self.wl_arr[self.idx]))
        return self

    def transform(self, X, **kwargs):
        return self.data[self.idx][X]