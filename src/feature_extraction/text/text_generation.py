import numpy as np
from ..window_slider import TwoWaysSlider, Segmentator
from ...timeseries_object import FastIrregularUTSObject, TimeSeriesObject
from tqdm import tqdm
from sklearn.base import TransformerMixin, BaseEstimator
from .count_words import merge_documents, count_words

_IRR_HANDLER_OPTIONS = ["#", "supp_interp"]
_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


class TextGeneration(TransformerMixin, BaseEstimator):
    def __init__(self, win=50, wl=4, direct_bow=True, alphabet_size=np.array([4]), quantity=np.array(["mean"]),
                 num_reduction=True, irr_handler="#",
                 index_based_paa=False, tol=6, mean_bp_dist="normal",
                 threshold=None, verbose=True):

        # if wl == 1:
            # raise ValueError("method not implemented for word length 1")

        # if quantity is None:
        #     quantity = ["mean"]
        # elif isinstance(quantity, str):
        #     quantity = [quantity]
        #
        # if alphabet_size is None:
        #     print("alph_size is none")
        #     if isinstance(quantity, (list, np.ndarray)):
        #         alphabet_size = [4] * len(quantity)
        #         assert len(alphabet_size) == 1
        #     else:
        #         raise ValueError("unknown type '%s' for quantity" % type(quantity))
        #
        # elif not isinstance(alphabet_size, (list, np.ndarray)):
        #     print("alph_size is integer")
        #     alphabet_size = [alphabet_size]
        #     assert len(alphabet_size) == 1
        #
        # assert len(alphabet_size) == 1
        #
        # alphabet_size = np.array(alphabet_size)
        #
        # quantity = np.array(quantity)
        #
        # if not isinstance(win, (int, float)):
        #     raise ValueError("window type '%s' cannot be processed" % type(win))
        # if not isinstance(wl, int):
        #     raise ValueError("Word length must be integer, "
        #                      "'%s' type mas provided" % type(wl))
        # if irr_handler not in _IRR_HANDLER_OPTIONS:
        #     raise ValueError("irr_handler '%s' unknown, options are:" % irr_handler,
        #                      _IRR_HANDLER_OPTIONS)
        #
        #
        # assert type(alphabet_size) == type(quantity)
        # assert len(alphabet_size) == len(quantity)

        self.quantity = quantity
        self.alphabet_size = alphabet_size
        self.win = win
        self.wl = wl
        self.num_reduction = num_reduction
        self.irr_handler = irr_handler
        self.index_based_paa = index_based_paa
        self.tol = tol
        self.mean_bp_dist = mean_bp_dist
        self.threshold = threshold
        self.verbose = verbose
        self.disabled = not self.verbose
        self.direct_bow = direct_bow


        # some necessary pointers
        self._prev_word = -1  # for numerosity reduction
        self._valid_chars = self.wl  # for counting valid characters in word
        self._bp = {}  # dict that contains the break points
        self._segmentator = Segmentator(self.win, self.wl, index_based=self.index_based_paa)
        self._alph_size_by_quantity = {x: y for x, y in zip(self.quantity, self.alphabet_size)}
        self._dropped_count = 0

    def _threshold(self):
        if self.threshold is None:
            return self.wl
        return self.threshold

    @property
    def alph_size(self):
        return self.alphabet_size.prod()

    @property
    def bop_size(self):
        a = self.alph_size
        if self.irr_handler == "#":
            a += 1
        return a ** self.wl

    def _word_offset(self, k):
        a = self.alph_size
        if self.irr_handler == "#":
            a += 1
        return a ** k

    def _num_reduction_cond(self, word):
        c1 = self.num_reduction
        c2 = self._prev_word != word
        return c1 and c2 or not c1

    def _cond_to_word(self, i, j):
        return j-i >= max(1, self.tol)

    def _cond_to_char(self, i, j):
        return j - i >= max(1, self.tol // self.wl)

    def _interpolate_new_timeseries(self, ts, i, j, ini_seq, empty_segments):
        interp_func = ts.interp1d(i, j)
        new_times = ts.times[i:j]
        for k in empty_segments:
            if k == 0 or k == self.wl - 1:
                continue
            ini = ini_seq + k * self._segmentator.sub_win
            end = ini + self._segmentator.sub_win
            if ini < ts.times[i] or end > ts.times[j-1]:
                continue
            new_times = np.append(new_times, np.linspace(ini, end, 3))
        new_times = np.sort(new_times)
        new_fluxes = interp_func(new_times)
        return FastIrregularUTSObject(new_fluxes, new_times)

    def _interp_condition(self, empty_segments):
        c1 = len(empty_segments) > 0
        c2 = "interp" in self.irr_handler
        c3 = any([x not in [0, self.wl - 1] for x in empty_segments])
        return c1 and c2 and c3

    def _set_global_break_points(self, ts: FastIrregularUTSObject):
        i = 0
        j = len(ts.fluxes)
        for k, v in self._alph_size_by_quantity.items():
            self._bp[k] = getattr(ts, "%s_break_points" % k)(
                i, j, v, dist=self.mean_bp_dist
            )

    def _set_local_break_points(self, ts, i, j, keys):
        for k in keys:
            v = self._alph_size_by_quantity[k]
            self._bp[k] = getattr(ts, "%s_break_points" % k)(
                i, j, v, dist=self.mean_bp_dist
            )

    def _compute_mean(self, *args):
        mean = args[0].paa_value(args[5], args[6], args[1], args[2])
        return np.digitize(mean, self._bp["mean"])

    def _compute_trend(self, *args):
        trend = args[0].trend_value(args[5], args[6])
        if trend is None:
            return
        return np.digitize(trend, self._bp["trend"])

    def _compute_min_max(self, *args):
        min_max = args[0].min_max_value(args[5], args[6])
        return np.digitize(min_max, self._bp["min_max"])

    def _compute_std(self, *args):
        std = args[0].std_value(args[5], args[6])
        if std is None:
            return None
        return np.digitize(std, self._bp["std"])

    def _transform(self, ts: FastIrregularUTSObject, slider: TwoWaysSlider):
        n = slider.n()
        doc = []
        dropped_count = 0

        if self.wl == 1:
            self._set_global_break_points(ts)

        self._prev_word = -1
        for k in range(n):
            i, j, ini, end = slider.get_subsequence(k)
            # if self._verbose:
            #     print("::> slider sub-sequence:", i, j, ini, end)
            if self._cond_to_word(i, j):
                word, empty_segments, segments = self._sequence_to_word(ts, i, j, ini)
                # print(i, j, word, empty_segments, segments)
                if len(word) > 0:
                    wordp = self._process_word(word, empty_segments)
                    if self._num_reduction_cond(wordp[0]):
                        # if self._verbose:
                        #     print("    resulting words:", wordp)
                        for w in wordp:
                            if w > self.bop_size:
                                raise ValueError("out of limits")
                            doc.append(w)
                        self._prev_word = wordp[0]
                    else:
                        # if self.verbose:
                        #     print("::> slider sub-sequence:", i, j, round(ini, 2),
                        #           " - dropped due to numerosity reduction")
                        pass
                else:
                    # if self.verbose:
                    #     print("::> slider sub-sequence:", i, j, round(ini, 2),
                    #           " - dropped due to empty segments")
                    dropped_count += 1
        return doc, dropped_count

    def _sequence_to_word(self, ts: FastIrregularUTSObject, i, j, ini_seq):
        segments, empty_segments = self._segmentator.segmentate(ts, i, j, ini_seq)
        # print(i, j, ini_seq, "segments:", list(segments))
        # if self._verbose:
        #     print("    empty_segments:", empty_segments)
        l_e = len(empty_segments)
        if self.wl - self._threshold() < l_e:
            # print("dropped")
            return [], empty_segments, segments
        if self._interp_condition(empty_segments):
            # print("interpolating")
            new_ts = self._interpolate_new_timeseries(ts, i, j, ini_seq, empty_segments)
            segments, empty_segments = self._segmentator.segmentate(new_ts, 0, j-i, ini_seq)
            # print("new segments:", segments)
            if len(empty_segments) > 3:
                # print("dropped")
                return [], [], []
            word = self._generate_word(new_ts, 0, j-i, segments)
        else:
            word = self._generate_word(ts, i, j, segments)

        return word, empty_segments, segments

    def _generate_word(self, ts, i, j, segments):
        if self.wl > 1:
            self._set_local_break_points(ts, i, j, self.quantity)
            meanx, sigmax = ts.mean_sigma_segment(i, j)
        else:
            meanx, sigmax = ts.mean_sigma_segment(0, ts._n)
        word = []
        self._valid_chars = self.wl
        for k in range(self.wl):
            ini, end = segments[k]
            val = self._segment_to_char(ts, meanx, sigmax, i, j, ini, end)
            word.append(self._word_offset(k) * val)
        return word

    def _segment_to_char(self, *args):
        if self._cond_to_char(args[5], args[6]):
            val = 0
            weigth = 1
            nn = len(self.quantity)
            for q in range(nn):
                inn_val = getattr(self, "_compute_" + self.quantity[nn - q - 1])(*args, q)
                if inn_val is None:
                    # if any of the computed quantities fail, the segment fail
                    self._valid_chars -= 1
                    return -1
                val += weigth * inn_val
                weigth *= self.alphabet_size[nn - q - 1]

        else:
            val = -1
            self._valid_chars -= 1
        return val

    def _process_word(self, word, empty_segments):
        if self.irr_handler == "#":
            wordp = [self._special_character_word(word)]
        elif self.irr_handler == "supp_interp":
            word = self._superposition_word(word, empty_segments)
            wordp = [np.sum(w) for w in word]
        else:
            wordp = [np.sum(word)]
        return wordp

    def _special_character_word(self, word):
        for i in range(self.wl):
            if word[i] < 0:
                word[i] = word[i] * -self.alph_size
        return np.sum(word)

    def _superposition_word(self, word: list, empty_segments):
        new_word = [np.array(word)]
        for es in empty_segments:
            new_word = self._multiplicate_word(new_word, es)
        return new_word

    def _multiplicate_word(self, word: list, i):
        new_word = []
        for word_i in word:
            base = word_i[i]
            for k in range(self.alph_size):
                word_i[i] = word_i[i] * -k
                new_word.append(word_i.copy())
                word_i[i] = base
        return new_word

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, dataset: np.ndarray, **transform_params):
        # print("SELF INFO:")
        # print(self.__dict__)

        n = len(dataset)
        corpus = np.full(n, None, dtype=object)
        slider = TwoWaysSlider(self.win, tol=self.tol)
        faileds = 0
        for i in tqdm(
                range(n),
                desc="[win: %f.3, wl: %d]" % (self.win, self.wl),
                dynamic_ncols=True, disable=self.disabled):

            doc, self._dropped_count = self.transform_object(dataset[i], slider)
            if doc is not None:
                corpus[i] = doc
            else:
                faileds += 1

        if self.direct_bow:
            new_x = merge_documents(corpus, _BANDS, self.bop_size)
            return count_words(new_x, self.bop_size * len(_BANDS))
        else:
            return corpus

    def transform_object(self, ts_obj, slider: TwoWaysSlider):
        if isinstance(ts_obj, TimeSeriesObject):
            ts_obj = ts_obj.to_fast_irregular_uts_object(_BANDS)

        doc_mb = {}
        dropped_mb = 0
        all_none = True
        for k, v in ts_obj.items():
            # print("BAND ", k)
            if v is not None:
                # if self.verbose:
                #     print("BAND:", k)
                slider.fit(v.times)
                doc, dropped = self._transform(v, slider)
                # print(dropped)
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