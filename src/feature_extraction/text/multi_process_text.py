from tqdm.contrib.concurrent import process_map
from sklearn.base import TransformerMixin, BaseEstimator

from .text_generation import TextGeneration
from ..window_slider import TwoWaysSlider
from .count_words import merge_documents, count_words, multivariate_count_words_flattened
import numpy as np


class MPTextGenerator(TransformerMixin, BaseEstimator):
    def __init__(self, bands=None, n_jobs=6, direct_bow=True, opt_desc="", **doc_gen_kwargs):
        if bands is None:
            raise ValueError("need to define the bands keywords")
        self.bands = bands

        self._win = doc_gen_kwargs.pop("win", None)
        if self._win is None:
            raise ValueError("need to define a window")

        self._wl = doc_gen_kwargs.pop("wl", None)
        if self._wl is None:
            raise ValueError("need to define a word length")

        self._tol = doc_gen_kwargs.get("tol", 6)
        self.doc_kwargs = doc_gen_kwargs
        self.n_jobs = n_jobs
        self._direct_bow = direct_bow
        self._opt_desc = opt_desc

    def get_bop_size(self):
        doc_gen = TextGeneration(win=self._win, wl=self._wl, direct_bow=self._direct_bow, **self.doc_kwargs)
        return doc_gen.bop_size

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):

        r = process_map(self.transform_object, X, max_workers=self.n_jobs,
                        desc="[win: %.3f, wl: %d%s]" % (self._win, self._wl, self._opt_desc), chunksize=8)
        if self._direct_bow:
            bop_size = self.get_bop_size()
            new_x = merge_documents(np.array(r), self.bands, bop_size)
            return count_words(new_x, bop_size * len(self.bands))
        else:
            return r

    def transform_object(self, x):
        doc_gen = TextGeneration(self._win, self._wl, **self.doc_kwargs)
        slider = TwoWaysSlider(self._win, tol=self._tol)
        doc_mb, dropped_mb = doc_gen.transform_object(x, slider)
        return doc_mb


class MPTextGeneratorMultivariateCountWords(MPTextGenerator):
    def transform(self, X, **kwargs):

        r = process_map(self.transform_object, X, max_workers=self.n_jobs,
                        desc="[win: %.3f, wl: %d%s]" % (self._win, self._wl, self._opt_desc), chunksize=8)
        if self._direct_bow:
            bop_size = self.get_bop_size()
            return multivariate_count_words_flattened(r, self.bands, bop_size)
        else:
            return r