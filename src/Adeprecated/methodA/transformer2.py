from src.utils import AbstractCore
import copy
import string
import numpy as np
from src.timeseries_object import TimeSeriesObject
from .representation import BOPSparseRepresentation
from scipy.stats import norm, linregress


_VALID_KWARGS = {
    "special_character": True,
    "strategy": "special1",
    "verbose": False,
    "tol": 3,
    "alph_size": 4,
    "trend_alph_size": 4,
    "window": None,
    "word_length": None,
    "feature": "mean",
    "numerosity_reduction": True,
    "threshold1": None,
}

_band_map = {
        0: 'lsstu',
        1: 'lsstg',
        2: 'lsstr',
        3: 'lssti',
        4: 'lsstz',
        5: 'lssty',
    }


def get_full_alphabet(max_alph_size):
    letters = list(string.ascii_letters)
    if max_alph_size > len(letters):
        n = max_alph_size // len(letters) + 1
        numbers = np.arange(n) + 1
        alphabet = [str(numbers[i]) + x for i in range(n) for x in letters]
    else:
        alphabet = letters
    return alphabet


class BOPTransformer(AbstractCore):

    def get_valid_kwargs(self) -> dict:
        return copy.deepcopy(_VALID_KWARGS)

    @classmethod
    def module_name(cls):
        return "MethodA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_bp = None
        self.trend_bp = self._trend_bp()
        self.pword = -1
        self.i = 0
        self.j = 0
        self.alphabet = self._set_alphabet()
        self._format = "csr"

    def _reset(self):
        self.pword = -1
        self.i = 0
        self.j = 0

    def _set_alphabet(self):
        alph_size = self._alph_size()
        full_alph = get_full_alphabet(alph_size)
        return full_alph[:alph_size]

    def _mean_bp(self, flux):
        points = np.linspace(0, 1, self["alph_size"] + 1)[1:-1]
        if self["word_length"] == 1:
            if len(flux) == 0:
                raise ValueError("empty flux arr")
            mean, std = np.mean(flux), np.std(flux)
        else:
            mean, std = 0, 1
        return norm.ppf(points, mean, std)

    def _trend_bp(self):
        ini = np.pi/2
        end = -np.pi/2
        return np.linspace(ini, end, self["trend_alph_size"] + 1)[1:-1]

    def _cumsum(self, data, n):
        cum1 = np.zeros(n + 1)
        cum2 = np.zeros(n + 1)
        psum = 0
        psum2 = 0
        for j in range(n):
            psum += data[j]
            psum2 += data[j] ** 2
            cum1[j+1] = psum
            cum2[j+1] = psum2
        return cum1, cum2

    def is_trend_value(self):
        return self["feature"] == "trend_value"

    def is_mean(self):
        return self["feature"] == "mean"

    def use_special_character(self):
        return self["special_character"]

    def _alph_size(self):
        alph_size = self["alph_size"]
        if self.is_trend_value():
            alph_size *= self["trend_alph_size"]
        return int(alph_size)

    def _bop_size(self):
        alph_size = self._alph_size()
        if self.use_special_character():
            alph_size += 1
        return int(alph_size ** self["word_length"])

    def _numerosity_reduction_condition(self, wordp):
        return (self["numerosity_reduction"] and self.pword != wordp) or not self["numerosity_reduction"]

    def _bop(self, flux, mjd, n) -> (BOPSparseRepresentation, str):
        bop_size = self._bop_size()
        cum1, cum2 = self._cumsum(flux, n)
        vector = np.zeros(bop_size)
        doc = ""
        self.i = 0
        self.j = 0
        self.pword = -1
        self.mean_bp = self._mean_bp(flux)
        while self.j < n:
            seq_i, seq_j = self._next_sequence(mjd, n)
            if seq_i == seq_j and seq_i == -1:
                break
            wordp, word = self._sequence_to_word(flux, mjd, n, cum1, cum2, seq_i, seq_j)
            wordp, word, valid = self._word_cases(wordp, word)
            if valid:
                if self._numerosity_reduction_condition(wordp):
                    vector[wordp] += 1
                    doc += word
                    self.pword = wordp
        ret = BOPSparseRepresentation(_format=self._format)
        ret.store_repr(vector)
        return ret, doc

    def _next_sequence(self, time_stamps: np.ndarray, n: int):
        while self.j < n:
            if time_stamps[self.j] - time_stamps[self.i] < self["window"]:
                self.j += 1
            else:
                if self.j - self.i - 1 > self["tol"]:
                    i = self.i
                    j = self.j
                    self.i += 1
                    return i, j
                else:
                    self.i += 1
                    self.j += 1
        if self.j - self.i - 1 < self["tol"]:
            return -1, -1
        else:
            return self.i, self.j

    def _sequence_to_word(self, flux, mjd, n, cum1: np.ndarray,
                          cum2: np.ndarray, seq_i: int, seq_j: int):
        if seq_i == seq_j:
            self.logger.warning("seq_j == seq_i encountered in 'mean' feature word. Word skipped")
            return
        if self["word_length"] == 1:
            return self._single_char_word(flux, mjd, cum1, seq_i, seq_j)

        sumx = cum1[seq_j] - cum1[seq_i]
        sumx2 = cum2[seq_j] - cum2[seq_i]
        meanx = sumx / (seq_j - seq_i)
        meanx2 = sumx2 / (seq_j - seq_i)
        sigmax = np.sqrt(np.abs(meanx2 ** 2 - meanx ** 2))
        seq_seg_j = seq_i
        seq_seg_win = 1.0 * self["window"] / self["word_length"]
        wordp = 0
        word = ""
        alph_size = self._alph_size()
        if self.use_special_character():
            alph_size += 1
        for w_i in range(self["word_length"]):
            seq_seg_i = seq_seg_j
            seq_seg_j = self._next_segment(mjd, n, w_i, seq_seg_win, seq_i, seq_seg_j)
            if seq_seg_j - seq_seg_i <= max(self["tol"] // self["word_length"], 1):
                val = alph_size - 1
                char = "#"
            else:
                sumsub = cum1[seq_seg_j] - cum1[seq_seg_i]
                avgsub = sumsub / (seq_seg_j - seq_seg_i)
                if sigmax > 0:
                    paa = (avgsub - meanx) / sigmax
                    val = np.digitize(paa, self.mean_bp)
                else:
                    val = np.digitize(0, self.mean_bp)

                if self.is_trend_value() and seq_seg_j - seq_seg_i > 1:
                    slope = self._slope(flux, mjd, seq_seg_i, seq_seg_j)
                    trend = np.arctan(slope)
                    slope_val = np.digitize(trend, self.trend_bp)
                    val = val * self["trend_alph_size"] + slope_val
                char = self.alphabet[val]
            wordp += (alph_size ** w_i) * val
            word += char
        return wordp, word

    def _next_segment(self, mjd, n, w_i: int,
                      seq_seg_win: float, seq_i: int, seq_seg_j: int):
        ini_time = mjd[seq_i]
        cmp = seq_seg_win * (w_i + 1) + ini_time
        while seq_seg_j < n and mjd[seq_seg_j] < cmp:
            if seq_seg_j == n:
                break
            seq_seg_j += 1
        return seq_seg_j

    def _word_cases(self, wordp: int, word: str):
        count_special = word.count("#")
        if self.use_special_character():
            if self["strategy"] == "special1":
                # allow a maximum number of special characters
                threshold = self["threshold1"]
                if threshold is None:
                    threshold = max(0, (len(word) // 2) - 1)
                valid = count_special <= threshold
            else:
                raise ValueError("strategy '%s' unknown" % self["strategy"])
        else:
            valid = count_special == 0
        return wordp, word, valid

    def _single_char_word(self, flux, mjd, cum1: np.ndarray,
                          i: int, j: int):
        sumx = cum1[j] - cum1[i]
        meanx = sumx / (j - i)
        alph_size = self._alph_size()
        if j - i > self['tol']:
            val = np.digitize(meanx, self.mean_bp)
            if self.is_trend_value():
                slope = self._slope(flux, mjd, i, j)
                trend = np.arctan(slope)
                slope_val = np.digitize(trend, self.trend_bp)
                val = val * self["trend_alph_size"] + slope_val

            char = self.alphabet[val]
        else:
            val = alph_size
            char = "#"
        return val, char

    def _slope(self, flux, mjd, i: int, j: int):
        if i > j:
            raise ValueError("i cannot be higher than j")
        if j - i <= 1:
            self.logger.warning("slope cannot be computed with 1 value. Set to 0")
            return 0
        if all(flux[x] == flux[i] for x in range(i,j)):
            self.logger.warning("all flux values are the same, slope value set to 0.")
            return 0
        if all(mjd[x] == mjd[i] for x in range(i,j)):
            self.logger.warning("all times are the same, slope value set to inf.")
            return np.inf

        slope, _, _, _, _ = linregress(mjd[i:j], flux[i:j])
        if np.isnan(slope):
            self.logger.warning("slope value {} is bad, times={}, flux={}".format(slope, mjd[i:j], flux[i:j]))
            slope = 0
        return slope

    def count_failed(self, vector) -> int:
        sums = np.sum(vector, axis=1)
        failed = len(np.where(sums == 0)[0])
        if failed > 0:
            s = "[COUNT FAILED] {} time series couldnt be".format(failed)
            s += " represented for [win={},wl={}]".format(self["window"],
                                                          self["word_length"])
            self.logger.warning(s)
        return failed

    def transform_dataset(self, dataset: np.ndarray) -> (BOPSparseRepresentation, int):
        matrix = BOPSparseRepresentation(_format=self._format)
        empty_vec = np.zeros(self._bop_size())
        for i, ts in enumerate(dataset):
            observations = ts.observations
            observations = observations.sort_values(by=["time"])
            ret = BOPSparseRepresentation(_format=self._format)
            for band_id, band_key in _band_map.items():
                tmp = observations[observations["band"] == band_key]
                if len(tmp) == 0:
                    self.logger.warning("Empty band, skipping and setting repr to {} zeros".format(self._bop_size()))
                    ret_tmp = BOPSparseRepresentation(_format=self._format)
                    ret_tmp.store_repr(empty_vec)
                else:
                    ret_tmp, doc = self._bop(tmp["flux"].to_numpy(), tmp["time"].to_numpy(), tmp.shape[0])

                if ret.vector is None:
                    ret.copy_from(ret_tmp)
                else:
                    ret.hstack_repr(ret_tmp)
            if matrix.vector is None:
                matrix.copy_from(ret)
            else:
                matrix.vstack_repr(ret)

        failed = self.count_failed( matrix.to_array())
        return matrix, failed
