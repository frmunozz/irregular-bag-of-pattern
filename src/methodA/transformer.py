from ..utils import AbstractCore
import copy
import numpy as np
from scipy import sparse
import string
from scipy.stats import norm
import multiprocessing as mp
import queue
from .cv_funtions import cv_fea_num_finder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import norm, linregress


#  import warnings
# warnings.filterwarnings('error')


_VALID_KWARGS = {
    "special_character": True,
    "strategy": "special1",
    "verbose": True,
    "tol": 1,
    "alph_size": 4,
    "trend_alph_size": 4,
    "window": None,
    "word_length": None,
    "full_alphabet": None,
    "feature": "mean",
    "numerosity_reduction": True,
    "threshold1": None,
}


def get_full_alphabet(max_alph_size):
    letters = list(string.ascii_letters)
    if max_alph_size > len(letters):
        N = max_alph_size // len(letters) + 1
        numbers = np.arange(N) + 1
        alphabet = [str(numbers[i]) + x for i in range(N) for x in letters]
    else:
        alphabet = letters
    return alphabet


class CountVectorizer(AbstractCore):

    @classmethod
    def module_name(cls):
        return "MethodA"

    def get_valid_kwargs(self) -> dict:
        return copy.deepcopy(_VALID_KWARGS)

    def __init__(self, **kwargs):
        super(CountVectorizer, self).__init__(**kwargs)
        self._wl = None
        self._win = None
        self.pword = -1
        self.count_vector = None
        self.corpus = []
        self.alphabet = []
        self.mean_bp = norm.ppf(np.linspace(0, 1, self["alph_size"] + 1)[1:-1])
        self.trend_bp = np.linspace(-np.pi/4, np.pi/4, self["trend_alph_size"] + 1)[1:-1]

    def clear(self):
        self._wl = None
        self._win = None
        self.pword = -1
        self.count_vector = None
        self.corpus = []
        self.alphabet = []
        self.mean_bp = norm.ppf(np.linspace(0, 1, self["alph_size"] + 1)[1:-1])

    def cumsum(self, dataset, m):
        cum1 = []
        cum2 = []
        psum = np.zeros(m)
        psum2 = np.zeros(m)
        for i in range(m):
            data = dataset[i]
            n = data.size
            cum1_data = np.zeros(n+1)
            cum2_data = np.zeros(n+1)
            for j in range(n):
                psum[i] += data[j]
                psum2[i] += data[j] ** 2
                cum1_data[j+1] = psum[i]
                cum2_data[j+1] = psum2[i]
            cum1.append(cum1_data)
            cum2.append(cum2_data)

        return cum1, cum2

    def get_bop_size(self, wl):
        if self["feature"] == "trend_value":
            alph_size = self["alph_size"] * self["trend_alph_size"]
        else:
            alph_size = self["alph_size"]

        if self["special_character"]:
            alph_size += 1

        return alph_size ** wl

    def get_alph_size(self):
        if self["feature"] == "trend_value":
            return self["alph_size"] * self["trend_alph_size"]
        else:
            return self["alph_size"]

    def bop(self, dataset, times, wl, win):
        self._wl = wl
        self._win = win
        bop_size = self.get_bop_size(wl)
        m = len(dataset)
        cum1, cum2 = self.cumsum(dataset, m)
        self.count_vector = np.zeros((m, bop_size))
        self.corpus = [""] * m
        for k in range(m):
            self.pword = -1
            data = dataset[k]
            time_stamps = times[k]
            n = len(data)
            i = 0
            j = 0
            window = win
            if wl == 1:
                self.mean_bp = norm.ppf(np.linspace(0, 1, self["alph_size"] + 1)[1:-1],
                                        np.mean(data), np.std(data))
            while j < n:
                seq_i, seq_j, i, j = self._bop_get_next_sequence(time_stamps, n, i, j, window)
                if seq_i == seq_j and seq_i == -1:
                    break
                self.sequence_to_word(data, time_stamps, cum1[k], cum2[k], seq_i, seq_j, window, wl, k, n)

    def _bop_get_next_sequence(self, time_stamps, n, i, j, window):
        while j < n:
            if time_stamps[j] - time_stamps[i] < window:
                j += 1
            else:
                if j - i - 1 > self["tol"]:
                    seq_ini = i
                    seq_end = j
                    i += 1
                    return seq_ini, seq_end, i, j
                else:
                    i += 1
                    j += 1
        if j - i - 1 < self["tol"]:
            return -1, -1, i, j
        else:
            return i, j, i, j

    def sequence_to_word(self, data, time_stamps, cum1, cum2, i, j, window, wl, k, n):
        if self["feature"] == "mean":
            self.mean_feature_word(time_stamps, cum1, cum2, i, j, window, wl, k, n)
        elif self["feature"] == "trend_value":
            self.trend_value_feature_word(data, time_stamps, cum1, cum2, i, j, window, wl, k, n)
        else:
            raise ValueError("feature '%s' unknown" % self["feature"])

    def trend_value_feature_word(self, data, time_stamps, cum1, cum2, i, j, window, wl, k, n):
        if j == i:
            self.logger.warning("j == i encountered in 'mean' feature word computation")
            return
        if wl==1:
            self.single_char_trend_value(data, time_stamps, cum1, i, j, k)
            return

        sumx = cum1[j] - cum1[i]
        sumx2 = cum2[j] - cum2[i]
        meanx = sumx / (j - i)
        sigmax = np.sqrt(np.abs(sumx2 / (j - i) - meanx * meanx))
        seg_j = i
        seg_window = window / wl
        wordp = 0
        word = ''

        alph_size = self["alph_size"] * self["trend_alph_size"]
        if self["special_character"]:
            alph_size += 1

        for w_i in range(wl):
            seg_i = seg_j
            seg_j = self._bop_get_next_segment(time_stamps, n, w_i, i, seg_j, seg_window)
            if seg_j - seg_i <= 1:
                trend_value = alph_size - 1
                char = "#"
            else:
                sumsub = cum1[seg_j] - cum1[seg_i]
                avgsub = sumsub / (seg_j - seg_i)
                if sigmax > 0:
                    paa = (avgsub - meanx) / sigmax
                    val = np.digitize(paa, self.mean_bp)
                else:
                    val = np.digitize(0, self.mean_bp)
                slope = self.get_slope(data, time_stamps, seg_i, seg_j)
                trend = np.arctan(slope)
                slope_val = np.digitize(trend, self.trend_bp)
                trend_value = val * self["trend_alph_size"] + slope_val
                # print(trend_value, val, slope_val)
                char = self.alphabet[trend_value]
                # print(val, char)
            wordp += (alph_size ** w_i)*trend_value
            word += char
        self.word_feature_cases(wordp, word, k)

    def mean_feature_word(self, time_stamps, cum1, cum2, i, j, window, wl, k, n):
        if j == i:
            self.logger.warning("j == i encountered in 'mean' feature word computation")
            return
        if wl == 1:
            self.single_char_mean(cum1, i, j, k)
            return

        sumx = cum1[j] - cum1[i]
        sumx2 = cum2[j] - cum2[i]
        meanx = sumx / (j-i)
        sigmax = np.sqrt(np.abs(sumx2 / (j-i) - meanx * meanx))
        seg_j = i
        seg_window = window / wl
        wordp = 0
        word = ''
        alph_size = self["alph_size"]
        if self["special_character"]:
            alph_size += 1

        for w_i in range(wl):
            seg_i = seg_j
            seg_j = self._bop_get_next_segment(time_stamps, n, w_i, i, seg_j, seg_window)
            if seg_j - seg_i <= self["tol"] // wl:
                val = self["alph_size"]
                char = "#"
            else:
                sumsub = cum1[seg_j] - cum1[seg_i]
                avgsub = sumsub / (seg_j - seg_i)
                if sigmax > 0:
                    paa = (avgsub - meanx) / sigmax
                    val = np.digitize(paa, self.mean_bp)
                else:
                    val = np.digitize(0, self.mean_bp)
                char = self.alphabet[val]
                # print(val, char)
            wordp += (alph_size ** w_i)*val
            word += char
        self.word_feature_cases(wordp, word, k)

    def _bop_get_next_segment(self, time_stamps, n, w_i, i, seg_j, seg_window):
        ini_time = time_stamps[i]
        cmp = seg_window * (w_i + 1) + ini_time
        while seg_j < n and time_stamps[seg_j] < cmp:
            if seg_j == n:
                break
            seg_j += 1
        return seg_j

    def word_feature_cases(self, wordp, word, k):
        count_special = word.count('#')
        if self["special_character"]:
            if self["strategy"] == "special1":
                # allow a maximum number of special characters
                threshold = max(0, len(word) // 2 - 1) if self["threshold1"] is None else self["threshold1"]
                if count_special <= threshold:
                    self.add_word(wordp, word, k)
            else:
                raise ValueError("strategy '%s' unknown" % self["strategy"])
        else:
            if count_special == 0:
                self.add_word(wordp, word, k)

    def add_word(self, wordp, word, k):
        if self["numerosity_reduction"]:
            if self.pword != wordp:
                self.count_vector[k][wordp] += 1
                self.pword = wordp
                if len(self.corpus[k]) > 0:
                    self.corpus[k] += ' '
                self.corpus[k] += word
        else:
            self.count_vector[k][wordp] += 1
            if len(self.corpus[k]) > 0:
                self.corpus[k] += ' '
            self.corpus[k] += word

    def single_char_mean(self, cum1, i, j, k):
        sumx = cum1[j] - cum1[i]
        meanx = sumx / (j-i)
        if j - i > self['tol']:
            val = np.digitize(meanx, self.mean_bp)
            char = self.alphabet[val]
        else:
            val = self["alph_size"]
            char = "#"
        self.word_feature_cases(val, char, k)

    def single_char_trend_value(self, data, time_stamps, cum1, i, j, k):
        sumx = cum1[j] - cum1[i]
        meanx = sumx / (j - i)
        if j - i > self['tol']:
            val = np.digitize(meanx, self.mean_bp)
            slope = self.get_slope(data, time_stamps, i, j)
            trend = np.arctan(slope)
            slope_val = np.digitize(trend, self.trend_bp)
            trend_value = val * self["trend_alph_size"] + slope_val
            char = self.alphabet[trend_value]
        else:
            trend_value = self.get_alph_size()
            char = "#"
        self.word_feature_cases(trend_value, char, k)

    def get_slope(self, data, time_stamps, i, j):
        # print(i, j)
        if j-i == 1:
            self.logger.warning("slope cannot be computed with 1 value, setting to 0")
            return 0
        slope, _, _, _, _ = linregress(time_stamps[i:j], data[i:j])
        if np.isnan(slope) or np.isinf(slope):
            self.logger.warning("slope value {} is bad".format(slope))
        return slope

    def count_failed(self):
        sums = np.sum(self.count_vector, axis=1)
        failed = len(np.where(sums == 0)[0])
        if failed > 0:
            self.logger.warning("[COUNT FAILED] {} time series couldnt be represented for [win={},wl={}]".format(failed,
                                                                                           self._win,
                                                                                           self._wl))
        return failed


class BOWTransformer(CountVectorizer):
    def fit(self, times):
        window = self["window"]
        word_length = self["word_length"]
        if window is None:
            window_widths = []
            for t in times:
                window_widths.append(t[-1]-t[0])
            max_window = np.mean(window_widths)
            window = [int(max_window / (2**i)) for i in range(6)]
        if isinstance(window, (int, float)):
            window = [window]
        if word_length is None:
            word_length = [1, 2, 4]
        if isinstance(word_length, int):
            word_length = [word_length]

        tot_combi = len(window) * len(word_length)
        self["window"] = window
        self["word_length"] = word_length
        self["full_alphabet"] = get_full_alphabet(self["alph_size"] * tot_combi)

    def transform(self, dataset, times):
        alph_offset = 0
        wl_arr = []
        win_arr = []
        corpus_arr = []
        count_vector_arr = []

        if self["verbose"]:
            self.logger.info("Transforming dataset of {} time series".format(len(dataset)))
            self.logger.info("Using windows:{}".format(self["window"]))
            self.logger.info("Using word_lengths: {}".format(self["word_length"]))
            self.logger.info("Using feature: {}".format(self["feature"]))
            self.logger.info("Using alph_size: {}".format(self["alph_size"]))

        for wl in self["word_length"]:
            for win in self["window"]:
                self.alphabet = self["full_alphabet"][alph_offset:alph_offset + self["alph_size"]]
                if self["verbose"]:
                    self.logger.info("computing BagOfPatter for window: {}, word_length: {}".format(win, wl))
                    self.logger.info("Using alphabet: {}".format(self.alphabet))
                self.bop(dataset, times, wl, win)
                sums = np.sum(self.count_vector, axis=1)
                failed = len(np.where(sums == 0)[0])
                if failed > 0:
                    self.logger.warning("{} time series couldn't be transformed for param [{},{}]".format(failed,
                                                                                                          wl,
                                                                                                          win))
                alph_offset += self["alph_size"]
                wl_arr.append(wl)
                win_arr.append(win)
                corpus_arr.append(np.copy(self.corpus))
                count_vector_arr.append(np.copy(self.count_vector))
        return wl_arr, win_arr, corpus_arr, count_vector_arr


class BOPFTransformer(CountVectorizer):
    def __init__(self, **kwargs):
        super(BOPFTransformer, self).__init__(**kwargs)
        self.best_centroid_vectors = None
        self.best_tf_idf_vectors = None

    def set_full_alphabet(self, n):
        self["full_alphabet"] = get_full_alphabet(n)

    def get_class_count(self, m, labels):
        classes = np.unique(labels)
        c = len(classes)
        class_count = np.zeros(c, dtype=int)
        positions = np.zeros(m, dtype=int)
        for i in range(m):
            pos = int(np.where(labels[i] == classes)[0][0])
            positions[i] = pos
            class_count[pos] += 1
        return classes, c, class_count, positions

    def anova(self, m, labels, wl):
        bop_size = self.get_bop_size(wl)
        classes, c, class_count, positions = self.get_class_count(m, labels)
        f_values = np.zeros(bop_size)
        for j in range(bop_size):
            sumall = 0.0
            sumc = np.zeros(c)
            for i in range(m):
                k = positions[i]
                sumall += self.count_vector[i][j]
                sumc[k] += self.count_vector[i][j]

            avgall = sumall / m
            ssa = 0.0
            avgc = np.zeros(c)
            for k in range(c):
                avgc[k] = sumc[k] / class_count[k]
                ssa += class_count[k] * (avgc[k] - avgall) * (avgc[k] - avgall)

            ssw = 0.0
            for i in range(m):
                k = positions[i]
                ssw += (self.count_vector[i][j] - avgc[k]) * (self.count_vector[i][j] - avgc[k])
            msa = 1.0 * ssa / (c - 1)
            msw = 1.0 * ssw / (m - c)
            if msw == 0 and msa != 0:
                f_values[j] = np.inf
            elif msa == 0 and msw == 0:
                f_values[j] = 0
            else:
                f_values[j] = np.round(msa / msw, 5)
            # self.logger.info("[ANOVA] f_value for j={} is {}".format(j, f_values[j]))

        return f_values, classes, c, class_count, positions

    def reduce_zeros(self, f_value, verbose=False):
        sort_idx = np.argsort(f_value)[::-1]
        bop_size = len(f_value)
        limit = np.argmax(f_value[sort_idx] == 0)
        if verbose:
            self.logger.info("[REDUCE ZEROS, win={}, wl={}]f_value size: {}, limit: {}, count_vect shape: {}".format(self._win, self._wl, bop_size, limit, self.count_vector.shape))
        sort_idx = sort_idx[:limit]
        # self.logger.info(("[REDUCE_ZEROS] shape count vector before={}".format(self.count_vector.shape)))
        self.count_vector = self.count_vector[:, sort_idx]
        # self.logger.info(("[REDUCE_ZEROS] shape count vector after={}".format(self.count_vector.shape)))
        # self.logger.info("[REDUCE ZEROS] representation size [{} -> {}] for [win={}, wl={}]".format(
        #     bop_size, limit, self._win, self._wl))
        return sort_idx, limit

    def cv_reduce_best_fea_num(self, limit, labels, prev_vectors, n_splits=5):
        best_fea_num_centroid = -1
        best_fea_num_tf_idf = -1
        best_acc_centroid = -1
        best_acc_tf_idf = -1
        for k in range(1, limit+1):
            stacked_vectors_centroid = self.stack_count_vectors(k, other_vector=prev_vectors["centroid"])
            stacked_vectors_tf_idf = self.stack_count_vectors(k, other_vector=prev_vectors["tf_idf"])
            # self.logger.info("[CV_REDUCER] k={}, shape_vec_c={}, shape_vec_tfidf={}, len_labels={}".format(k,
            #                                                                                 stacked_vectors_centroid.shape,
            #                                                                                 stacked_vectors_tf_idf.shape,
            #                                                                                                len(labels)))
            # self.logger.info("[CV_REDUCER] shape count_vector={}".format(self.count_vector.shape))
            # self.logger.info("[CV_REDUCER] type of stacked vectors: {}, {}".format(type(stacked_vectors_centroid),
            #                                                                        type(stacked_vectors_tf_idf)))

            bacc_centroid, bacc_tf_idf = cv_fea_num_finder((stacked_vectors_centroid, stacked_vectors_tf_idf),
                                                           labels, n_splits=n_splits)
            # self.logger.info("[CV_REDUCER] bacc_centroid: {}, bacc_tf_idf: {} [for k={}, wl={}, win={}]".format(
            #     bacc_centroid, bacc_tf_idf, k, self._wl, self._win))
            if best_acc_centroid <= bacc_centroid:
                best_acc_centroid = bacc_centroid
                best_fea_num_centroid = k
            if best_acc_tf_idf <= bacc_tf_idf:
                best_acc_tf_idf = bacc_tf_idf
                best_fea_num_tf_idf = k

        self.logger.info("[CV_REDUCER] best_bacc_centroid: {} for k: {} || best_bacc_tf_idf: {} for k: {} [win={}, wl={}]".format(
            best_acc_centroid, best_fea_num_centroid, best_acc_tf_idf, best_fea_num_tf_idf, self._win, self._wl))
        return best_fea_num_centroid, best_acc_centroid, best_fea_num_tf_idf, best_acc_tf_idf

    def cv_reduce_best_fea_num2(self, limit, labels, positions, class_count, y1in=None, y2in=None):
        best_fea_num_centroid = -1
        best_fea_num_tf_idf = -1
        best_acc_centroid = -1
        best_acc_tf_idf = -1

        m = len(labels)
        classes = np.unique(labels)
        c = len(classes)
        if y1in is None:
            y1 = np.zeros((m, c))
        else:
            y1 = np.copy(y1in)
        if y2in is None:
            y21 = np.zeros((m, c))
            y22 = np.zeros((m, c))
            y23 = np.zeros((m, c))
        else:
            y21 = np.copy(y2in[0])
            y22 = np.copy(y2in[1])
            y23 = np.copy(y2in[2])

        centroids = np.zeros((c, limit))
        tf_idfs = np.zeros((c, limit))

        for k in range(limit):
            for i in range(m):
                lbl = positions[i]
                centroids[lbl][k] += self.count_vector[i][k]
                tf_idfs[lbl][k] += self.count_vector[i][k]

            countc = 0.0
            for i in range(c):
                if tf_idfs[i][k] > 0:
                    countc += 1

            for i in range(c):
                centroids[i][k] = centroids[i][k] / class_count[i]
                if tf_idfs[i][k] > 0:
                    tf_idfs[i][k] = (1 + np.log10(tf_idfs[i][k]))*(np.log10(1 + c/countc))

            real = []
            pred_centroid = []
            pred_tf_idf = []
            for i in range(m):
                rmin = np.inf
                rmax = -np.inf
                lbl_centroid = -1
                lbl_tf_idf = -1
                lbl = positions[i]
                countc = class_count[lbl]
                for j in range(c):
                    r1 = y1[i][j]
                    pm = self.count_vector[i][k]
                    d1 = pm - centroids[j][k]
                    if j == lbl:
                        d1 += pm / countc
                    r1 += d1*d1
                    y1[i][j] = r1
                    if r1 < rmin:
                        rmin = r1
                        lbl_centroid = j

                    r21 = y21[i][j]
                    r22 = y22[i][j]
                    r23 = y23[i][j]

                    d2 = pm
                    if d2 > 0:
                        d2 = 1 + np.log10(d2)
                    r21 += d2 * tf_idfs[j][k]
                    r22 += d2*d2
                    r23 += tf_idfs[j][k] * tf_idfs[j][k]

                    y21[i][j] = r21
                    y22[i][j] = r22
                    y23[i][j] = r23
                    if r22 != 0 and r23 != 0:
                        r2 = r21*r21 / (r22*r23)
                    else:
                        r2 = 0
                    if r2 > rmax:
                        rmax = r2
                        lbl_tf_idf = j
                real.append(lbl)
                pred_centroid.append(lbl_centroid)
                pred_tf_idf.append(lbl_tf_idf)

            bacc_centroid = balanced_accuracy_score(real, pred_centroid)
            bacc_tf_idf = balanced_accuracy_score(real, pred_tf_idf)
            # if self._win == 26.59 and self._wl == 6:
            #     self.logger.info("[win={} wl={}] bacc centroid for k={} is {}".format(self._win, self._wl, k, bacc_centroid))

            if bacc_centroid >= best_acc_centroid:
                best_acc_centroid = bacc_centroid
                best_fea_num_centroid = k
            if bacc_tf_idf >= best_acc_tf_idf:
                best_acc_tf_idf = bacc_tf_idf
                best_fea_num_tf_idf = k

        return (best_fea_num_centroid, best_acc_centroid), (best_fea_num_tf_idf, best_acc_tf_idf), y1, (y21, y22, y23)

    def get_centroid_vectors(self, limit, labels, positions, class_count):
        m = len(labels)
        classes = np.unique(labels)
        c = len(classes)
        centroids = np.zeros((c, limit))
        for k in range(limit):
            for i in range(m):
                lbl = positions[i]
                centroids[lbl][k] += self.count_vector[i][k]

            for i in range(c):
                centroids[i][k] = centroids[i][k] / class_count[i]

        return centroids

    def cv_fea_num_centroid(self, limit, labels, positions, class_count, y1in=None):
        best_fea_num_centroid = -1
        best_acc_centroid = -1

        m = len(labels)
        classes = np.unique(labels)
        c = len(classes)
        y1 = np.zeros((m, c))
        if y1in is not None:
            y1 = y1in
        centroids = np.zeros((c, limit))

        y1out = np.zeros((m, c))

        for k in range(limit):
            for i in range(m):
                lbl = positions[i]
                centroids[lbl][k] += self.count_vector[i][k]

            for i in range(c):
                centroids[i][k] = centroids[i][k] / class_count[i]

            real = []
            pred_centroid = []
            for i in range(m):
                rmin = np.inf
                lbl_centroid = -1
                lbl = positions[i]
                countc = class_count[lbl]
                for j in range(c):
                    r1 = y1[i][j]
                    pm = self.count_vector[i][k]
                    d1 = pm - centroids[j][k]
                    if j == lbl:
                        d1 += pm / countc
                    r1 += d1 * d1
                    y1[i][j] = r1
                    if r1 < rmin:
                        rmin = r1
                        lbl_centroid = j


                real.append(lbl)
                pred_centroid.append(lbl_centroid)

            bacc_centroid = balanced_accuracy_score(real, pred_centroid)

            if bacc_centroid >= best_acc_centroid:
                best_acc_centroid = bacc_centroid
                best_fea_num_centroid = k
                y1out = np.copy(y1)

        return (best_fea_num_centroid, best_acc_centroid), y1out, centroids[:, :best_fea_num_centroid]

    def get_tf_idf_vectors(self, limit, labels, positions):
        m = len(labels)
        classes = np.unique(labels)
        c = len(classes)

        tf_idfs = np.zeros((c, limit))

        for k in range(limit):
            for i in range(m):
                lbl = positions[i]
                tf_idfs[lbl][k] += self.count_vector[i][k]

            countc = 0.0
            for i in range(c):
                if tf_idfs[i][k] > 0:
                    countc += 1

            for i in range(c):
                if tf_idfs[i][k] > 0:
                    tf_idfs[i][k] = (1 + np.log10(tf_idfs[i][k])) * (np.log10(1 + c / countc))

        return tf_idfs

    def prepare_count_vectors_for_tf_idf(self):
        n, m = self.count_vector.shape
        count_Vector_out = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if self.count_vector[i,j] > 0:
                    count_Vector_out[i,j] = 1 + np.log10(self.count_vector[i,j])
        return count_Vector_out

    def cv_fea_num_tf_idf(self, limit, labels, positions, y2in=None):
        best_fea_num_tf_idf = -1
        best_acc_tf_idf = -1

        m = len(labels)
        classes = np.unique(labels)
        c = len(classes)
        if y2in is None:
            y21 = np.zeros((m, c))
            y22 = np.zeros((m, c))
            y23 = np.zeros((m, c))
        else:
            y21 = np.copy(y2in[0])
            y22 = np.copy(y2in[1])
            y23 = np.copy(y2in[2])

        tf_idfs = np.zeros((c, limit))
        y21out = None
        y22out = None
        y23out = None

        for k in range(limit):
            for i in range(m):
                lbl = positions[i]
                tf_idfs[lbl][k] += self.count_vector[i][k]

            countc = 0.0
            for i in range(c):
                if tf_idfs[i][k] > 0:
                    countc += 1

            for i in range(c):
                if tf_idfs[i][k] > 0:
                    tf_idfs[i][k] = (1 + np.log10(tf_idfs[i][k])) * (np.log10(1 + c / countc))

            real = []
            pred_tf_idf = []
            for i in range(m):
                rmax = -np.inf
                lbl_tf_idf = -1
                lbl = positions[i]
                for j in range(c):
                    pm = self.count_vector[i][k]

                    r21 = y21[i][j]
                    r22 = y22[i][j]
                    r23 = y23[i][j]

                    d2 = pm
                    if d2 > 0:
                        d2 = 1 + np.log10(d2)
                    r21 += d2 * tf_idfs[j][k]
                    r22 += d2 * d2
                    r23 += tf_idfs[j][k] * tf_idfs[j][k]

                    y21[i][j] = r21
                    y22[i][j] = r22
                    y23[i][j] = r23
                    if r22 != 0 and r23 != 0:
                        r2 = r21 * r21 / (r22 * r23)
                    else:
                        r2 = 0
                    if r2 > rmax:
                        rmax = r2
                        lbl_tf_idf = j
                real.append(lbl)
                pred_tf_idf.append(lbl_tf_idf)

            bacc_tf_idf = balanced_accuracy_score(real, pred_tf_idf)
            # if self._win == 26.59 and self._wl == 6:
            #     self.logger.info("[win={} wl={}] bacc centroid for k={} is {}".format(self._win, self._wl, k, bacc_centroid))

            if bacc_tf_idf >= best_acc_tf_idf:
                best_acc_tf_idf = bacc_tf_idf
                best_fea_num_tf_idf = k
                y21out = np.copy(y21)
                y22out = np.copy(y22)
                y23out = np.copy(y23)


        return (best_fea_num_tf_idf, best_acc_tf_idf), (y21out, y22out, y23out)


        # y1 = np.zeros((m, c))
        # y2 = np.zeros((m, c))
        # class_vectors = np.zeros((c, limit))
        # class_count = np.zeros(c)
        # for k in range(m):
        #     lbl = np.where(classes == labels[k])[0][0]
        #     class_vectors[lbl] += self.count_vector[k]
        #     class_count[lbl] += 1
        #
        # for j in range(limit):
        #     for train_index, test_index in splitter.split(self.count_vector, labels):
        #         test_vectors = self.count_vector[test_index]
        #
        #         for u in test_index:
        #             lbl = np.where(classes == labels[u])[0][0]
        #             class_vectors[lbl] -= self.count_vector[k]
        #             class_count[lbl] -= 1
        #
        #         centroid = class_vectors / class_count
        #
        #         for i in range(c):
        #             r = y1[]
        #
        #         for u in test_index:
        #             lbl = np.where(classes == labels[u])[0][0]
        #             class_vectors[lbl] += self.count_vector[k]
        #             class_count[lbl] += 1

        # for i in range(m):

    # def stack_fea_centroid(self, k):
    #     if self.best_centroid_vectors is None:
    #         return sparse.lil_matrix(self.count_vector[:, :k])
    #     else:
    #         return sparse.hstack((self.best_centroid_vectors, self.count_vector[:, :k]), format="lil")
    #
    # def stack_fea_tf_idf(self, k):
    #     if self.best_tf_idf_vectors is None:
    #         return self.count_vector[:, :k]
    #     else:
    #         return sparse.hstack((self.best_tf_idf_vectors, self.count_vector[:, :k]), format="lil")

    def stack_count_vectors(self, k, other_vector=None):
        if other_vector is None:
            return sparse.lil_matrix(self.count_vector[:, :k])
        else:
            return sparse.hstack((other_vector, self.count_vector[:, :k]), format="lil")


def _transformer_worker(dataset, times, lock, comb_to_try, r_queue, **kwargs):
    try:
        transf = CountVectorizer(**kwargs)
        transf.fit(dataset, times)
        transf.logger.info("start countVectorizer on worker '%s'" % mp.current_process().name)
        while True:
            try:
                lock.acquire()
                win, wl, i = comb_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                full_alph = get_full_alphabet(transf["alph_size"])
                transf.alphabet = full_alph[:transf["alph_size"]]
                if transf["verbose"]:
                    transf.logger.info("computing BagOfPatter for [{}, {}] on worker '{}'".format(win, wl,
                                                                                                  mp.current_process().name))
                transf.bop(dataset, times, wl, win)
                sums = np.sum(transf.count_vector, axis=1)
                failed = len(np.where(sums == 0)[0])
                if failed > 0:
                    transf.logger.warning("{} time series couldnt be represented for [{},{}]".format(failed, wl, win))
                r_queue.put((wl, win, sparse.lil_matrix(transf.count_vector), i))
    except Exception as e:
        print("worker failed with error:", e)
        transf = None
    finally:
        # if transf is not None:
        #     transf.logger.info("worker '%' DONE" % mp.current_process().name)
        # else:
        print("worker '%s' DONE" % mp.current_process().name)


def transformer_mp(dataset, times, wl_arr, win_arr, n_process="default", **kwargs):

    if n_process == "default":
        n_process = mp.cpu_count()


    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(wl_arr) * len(win_arr)
    combinations_to_try = mp.Queue()

    i = 0
    for wl in wl_arr:
        for win in win_arr:
            combinations_to_try.put((win, wl, i))
            i += 1

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=_transformer_worker,
                       args=(dataset, times, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    wl_arr = [None] * n_combinations
    win_arr = [None] * n_combinations
    count_vec_arr = [None] * n_combinations
    num_res = result_queue.qsize()
    while num_res > 0:
        wl, win, count_vec, i = result_queue.get()
        wl_arr[i] = wl
        win_arr[i] = win
        count_vec_arr[i] = count_vec
        num_res -= 1

    return wl_arr, win_arr, count_vec_arr


def transformer_mp_tuples(dataset, times, tuples, n_process="default", **kwargs):

    if n_process == "default":
        n_process = mp.cpu_count()


    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(tuples)
    combinations_to_try = mp.Queue()

    i = 0
    for tuple in tuples:
        combinations_to_try.put((tuple[0], tuple[1], i))
        i += 1

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=_transformer_worker,
                       args=(dataset, times, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    wl_arr = [None] * n_combinations
    win_arr = [None] * n_combinations
    count_vec_arr = [None] * n_combinations
    num_res = result_queue.qsize()
    while num_res > 0:
        wl, win, count_vec, i = result_queue.get()
        wl_arr[i] = wl
        win_arr[i] = win
        count_vec_arr[i] = count_vec
        num_res -= 1

    return wl_arr, win_arr, count_vec_arr
