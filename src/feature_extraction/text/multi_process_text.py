from tqdm.contrib.concurrent import process_map
from sklearn.base import TransformerMixin, BaseEstimator

from .text_generation import TextGeneration
from ..window_slider import TwoWaysSlider


class MPTextGenerator(TransformerMixin, BaseEstimator):
    def __init__(self, bands=None, n_jobs=6, **doc_gen_kwargs):
        if bands is None:
            raise ValueError("need to define the bands keywords")
        self.bands = bands

        self._win = doc_gen_kwargs.pop("win", None)
        if self._win is None:
            raise ValueError("need to define a window")

        self._wl = doc_gen_kwargs.pop("word_length", None)
        if self._wl is None:
            raise ValueError("need to define a word length")

        self._tol = doc_gen_kwargs.get("tol", 6)
        self.doc_kwargs = doc_gen_kwargs
        self.n_jobs = n_jobs

    def get_bop_size(self):
        doc_gen = TextGeneration(self._win, **self.doc_kwargs)
        return doc_gen.bop_size

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):

        r = process_map(self.transform_object, X, max_workers=self.n_jobs,
                        desc="[win: %.3f, wl: %d]" % (self._win, self._wl))
        return r

    def transform_object(self, x):
        doc_gen = TextGeneration(self._win, self._wl, **self.doc_kwargs)
        slider = TwoWaysSlider(self._win, tol=self._tol)
        doc_mb, dropped_mb = doc_gen.transform_object(x, slider)
        return doc_mb