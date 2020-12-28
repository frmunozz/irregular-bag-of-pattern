import numpy as np
from ..segmentation import Slider, TwoWaysSlider, Segmentator
from src.timeseries_object import FastIrregularUTSObject, TimeSeriesObject
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict
import multiprocessing as mp
import pdb

_IRR_HANDLER_OPTIONS = ["special_character"]
_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


class DocumentGeneration(TransformerMixin):
    def __init__(self, window, word_length=2, alph_size=None, quantity=None,
                 num_reduction=True, irr_handler="special_character",
                 index_based_paa=False, tol=6, mean_bp_dist="normal",
                 threshold_missing=None, verbose=True):

        if quantity is None:
            quantity = ["mean"]
        elif isinstance(quantity, str):
            quantity = [quantity]

        if alph_size is None:
            if isinstance(quantity, str):
                alph_size = 4
            elif isinstance(quantity, (list, np.ndarray)):
                alph_size = [4] * len(quantity)
            else:
                raise ValueError("unknown type '%s' for quantity" % type(quantity))

        elif isinstance(alph_size, int):
            alph_size = [alph_size]

        alph_size = np.array(alph_size)

        quantity = np.array(quantity)

        if not isinstance(window, (int, float)):
            raise ValueError("window type '%s' cannot be processed" % type(window))
        if not isinstance(word_length, int):
            raise ValueError("Word length must be integer, "
                             "'%s' type mas provided" % type(word_length))
        if irr_handler not in _IRR_HANDLER_OPTIONS:
            raise ValueError("irr_handler '%s' unknown, options are:" % irr_handler,
                             _IRR_HANDLER_OPTIONS)

        assert type(alph_size) == type(quantity)
        assert len(alph_size) == len(quantity)

        self._quantity = quantity
        self._alph_size = alph_size
        self._win = window
        self._wl = word_length
        self._num_reduction = num_reduction
        self._irr_handler = irr_handler
        self._index_based_paa = index_based_paa
        self._tol = tol
        self._mean_bp_dist = mean_bp_dist
        self._threshold_missing = threshold_missing
        self._verbose = verbose

        # some necessary pointers
        self._prev_word = -1  # for numerosity reduction
        self._missing = 0  # for counting missing characters due to irregularity
        self._bp = {}  # dict that contains the break points
        self._segmentator = Segmentator(self._win, self._wl, index_based=self._index_based_paa)
        self._alph_size_by_quantity = {x: y for x, y in zip(self._quantity, self._alph_size)}

    @property
    def alph_size(self):
        return self._alph_size.prod()

    def use_special_character(self):
        return self._irr_handler == "special_character"

    @property
    def bop_size(self):
        if self.use_special_character():
            return (self.alph_size + 1) ** self._wl
        else:
            return self.alph_size ** self._wl

    def _word_offset(self, k):
        if self.use_special_character():
            return (self.alph_size + 1) ** k
        else:
            return self.alph_size ** k

    def _num_reduction_cond(self, word):
        c1 = self._num_reduction
        c2 = self._prev_word != word
        return c1 and c2 or not c1

    def _cond_to_word(self, i, j):
        return j-i >= max(1, self._tol)

    def _cond_to_char(self, i, j):
        return j - i >= max(1, self._tol // self._wl)

    def _transform(self, ts: FastIrregularUTSObject, slider: TwoWaysSlider):
        n = slider.n()
        doc = []
        dropped_count = 0

        self._prev_word = -1
        if self._wl == 1:
            self._set_global_break_points(ts)
        for k in range(n):
            i, j, ini, end = slider.get_subsequence(k)
            valid = False
            if self._cond_to_word(i, j):
                word = self._sequence_to_word(ts, i, j)
                valid = self._check_word(word)
                if valid and self._num_reduction_cond(word):
                    doc.append(word)
                    self._prev_word = word

            if not valid:
                dropped_count += 1
        return doc, dropped_count

    def _sequence_to_word(self, ts: FastIrregularUTSObject, i, j):
        self._set_local_break_points(ts, i, j, self._quantity)
        meanx, sigmax = ts.mean_sigma_segment(i, j)
        # pdb.set_trace()
        segments = self._segmentator.segmentate(ts, i, j)
        word = 0
        self._missing = 0
        for k in range(self._wl):
            ini, end = segments[k]
            val = self._segment_to_char(ts, meanx, sigmax, i, j, ini, end)
            word += self._word_offset(k) * val
        return word

    def _segment_to_char(self, *args):
        if self._cond_to_char(args[5], args[6]):
            val = 0
            weigth = 1
            nn = len(self._quantity)
            for q in range(nn):
                val += weigth * getattr(
                    self, "_compute_" + self._quantity[nn - q - 1])(*args, q)
                weigth *= self._alph_size[nn - q - 1]

        else:
            val = self.alph_size
            self._missing += 1
        return val

    def _compute_mean(self, *args):
        mean = args[0].paa_value(args[5], args[6], args[1], args[2])
        return np.digitize(mean, self._bp["mean"])

    def _compute_trend(self, *args):
        trend = args[0].trend_value(args[5], args[6])
        return np.digitize(trend, self._bp["trend"])

    def _compute_min_max(self, *args):
        min_max = args[0].min_max_value(args[5], args[6])
        return np.digitize(min_max, self._bp["min_max"])

    def _compute_std(self, *args):
        std = args[0].std_value(args[5], args[6])
        return np.digitize(std, self._bp["std"])

    def _set_global_break_points(self, ts: FastIrregularUTSObject):
        i = 0
        j = len(ts.fluxes)
        for k, v in self._alph_size_by_quantity.items():
            self._bp[k] = getattr(ts, "%s_break_points" % k)(
                i, j, v, dist=self._mean_bp_dist
            )

    def _set_local_break_points(self, ts, i, j, keys):
        for k in keys:
            v = self._alph_size_by_quantity[k]
            self._bp[k] = getattr(ts, "%s_break_points" % k)(
                i, j, v, dist=self._mean_bp_dist
            )

    def _check_word(self, word):
        if self.use_special_character():
            if self._irr_handler == "special_character":
                threshold = self._threshold_missing
                if threshold is None:
                    threshold = max(0, (self._wl // 2) - 1)
                return self._missing <= threshold
            else:
                raise ValueError("not implemented irregular handler '%s'" % self._irr_handler)
        else:
            return self._missing == 0

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, dataset: np.ndarray, **transform_params):
        n = len(dataset)
        corpus = np.full(n, None, dtype=object)
        slider = TwoWaysSlider(self._win, tol=self._tol)
        faileds = 0
        for i in tqdm(
                range(n),
                desc="[win: %f.3, wl: %d, faileds: %d]" % (self._win,
                                                           self._wl,
                                                           faileds),
                dynamic_ncols=True):

            doc, dropped_count = self.transform_object(dataset[i], slider)
            if doc is not None:
                corpus[i] = doc
            else:
                faileds += 1

        return corpus

    def transform_object(self, ts_obj, slider: TwoWaysSlider):
        if isinstance(ts_obj, TimeSeriesObject):
            ts_obj = ts_obj.to_fast_irregular_uts_object(_BANDS)

        doc_mb = {}
        dropped_mb = 0
        all_none = True
        for k, v in ts_obj.items():
            if v is not None:
                slider.fit(v.times)
                doc, dropped = self._transform(v, slider)
                doc_mb[k] = doc
                dropped_mb += dropped
                if len(doc) > 0:
                    all_none = False
            else:
                doc_mb[k] = None
        if all_none:
            doc_mb = None

        return doc_mb, dropped_mb

    def transform_uts(self, ts_obj, slider):
        slider.fit(ts_obj.times)
        doc, dropped = self._transform(ts_obj, slider)
        if len(doc) == 0:
            return None
        return doc

    def decode_document_to_words(self, doc, alphabets):
        pass


class MPDocumentGenerator(TransformerMixin):
    def __init__(self, bands, win, n_jobs=6, **doc_gen_kwargs):
        self.bands = bands
        self._win = win
        self._wl = doc_gen_kwargs.get("word_length", 4)
        self._tol = doc_gen_kwargs.get("tol", 6)
        self.doc_kwargs = doc_gen_kwargs
        self.n_jobs = n_jobs

    def get_bop_size(self):
        doc_gen = DocumentGeneration(self._win, **self.doc_kwargs)
        return doc_gen.bop_size

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):

        r = process_map(self.transform_object, X, max_workers=self.n_jobs,
                        desc="[win: %.3f, wl: %d]" % (self._win, self._wl))
        return r

    def transform_object(self, x):
        doc_gen = DocumentGeneration(self._win, **self.doc_kwargs)
        slider = TwoWaysSlider(self._win, tol=self._tol)
        doc_mb, dropped_mb = doc_gen.transform_object(x, slider)
        return doc_mb


class DocumentSelector(TransformerMixin, BaseEstimator):
    def __init__(self, idx=None, data=None, win_arr=None, wl_arr=None):
        if idx is None or data is None:
            raise ValueError("need to set idx and data")
        self.idx = idx
        self.data = data
        self.win_arr = win_arr
        self.wl_arr = wl_arr

    def fit(self, X, y=None, **kwargs):
        print("gettign the count words for win idx: %f.3 and wl idx: %d" % (
            self.win_arr[self.idx], self.wl_arr[self.idx]))
        return self

    def transform(self, X, **kwargs):
        return self.data[self.idx][X]