
import numpy as np
from .slidingwindow import SlidingWindow
from scipy.stats import norm
import pdb


def get_seq_mean_std(cum_sum_ts, cum_sum2_ts, i, j):
    # pdb.set_trace()
    sumx = (cum_sum_ts[j] - cum_sum_ts[i]) / (j - i)
    sumx2 = (cum_sum2_ts[j] - cum_sum2_ts[i]) / (j - i)
    sumx2 = np.sqrt(sumx2 - sumx * sumx)
    return sumx, sumx2


def get_seg_mean(cum_sum_ts, sumx, sumx2, seg_i, seg_j):
    seg_mean = (cum_sum_ts[seg_j] - cum_sum_ts[seg_i]) / (seg_j - seg_i)
    seg_mean = (seg_mean - sumx) / sumx2
    return seg_mean


def word_to_number(sax, alphabet_size):
    l = len(sax) - 1
    s = 0
    for i, w in enumerate(sax):
        s += w * alphabet_size ** (l-i)
    return s


def features_sax_size(alphabet_size):
    return 1 << (2 * alphabet_size)


class BagOfPattern(object):
    def __init__(self, discard_empty_segments=False, word_length=3, window_length=0.25, tol=5, **kwargs):
        self.alphabet_size = 4
        self.discard_empty_segments = discard_empty_segments
        self.word_length = word_length
        self.window_length = window_length
        self.seg_tol = tol
        self.seq_tol = self.seg_tol * self.word_length
        self.bow = []
        self.word_count = None
        self.prev_sax = None
        self.for_hist = []

    def paa_transform(self, ts, seq_i, time_window, cum_sum_ts, cum_sum2_ts, seq_mean, seq_std):
        # pdb.set_trace()
        seg_i = seq_i
        seg_j = seq_i
        ini_seq_time = ts.t[seq_i]
        paa = []
        has_empty_segment = False
        for word_i in range(self.word_length):
            while ts.t[seg_j] - ini_seq_time <= time_window * (word_i + 1):
                if seg_j == ts.size() - 1:
                    break
                seg_j += 1
            if seg_i == seg_j:
                # the segment is empty
                has_empty_segment = True
                mu = np.inf
            else:
                mu = get_seg_mean(cum_sum_ts, seq_mean, seq_std, seg_i, seg_j)
            # pdb.set_trace()
            paa.append(mu)
            self.for_hist.append(mu)
        return paa, has_empty_segment

    def sax_transform(self, paa, has_empty_segment, ts_i):
        break_points = [-0.67, 0, 0.67, np.inf]
        if not has_empty_segment or not self.discard_empty_segments:
            sax = np.digitize(paa, break_points)
            if self.prev_sax is not None and np.array_equal(self.prev_sax, sax):
                return self.prev_sax
            self.word_count[ts_i][word_to_number(sax, self.alphabet_size)] += 1
            self.bow.append(sax)
            return sax
        else:
            return np.array([])

    def transform_timeseries(self, ts_i, ts, cum_sum_ts, cum_sum2_ts):
        time_window = ts.bandwidth() * self.window_length
        sw = SlidingWindow(ts, self.window_length, tol=self.seq_tol)
        while True:
            seq_i, seq_j = sw.get_sequence()
            if seq_i == seq_j and seq_i == -1:
                break
            # pdb.set_trace()
            seq_mean, seq_std = get_seq_mean_std(cum_sum_ts, cum_sum2_ts, seq_i, seq_j)

            paa, has_empty_segment = self.paa_transform(ts, seq_i, time_window,
                                                        cum_sum_ts, cum_sum2_ts,
                                                        seq_mean, seq_std)

            sax = self.sax_transform(paa, has_empty_segment, ts_i)
            self.prev_sax = sax

    def transform_dataset(self, dataset, cum_sum, cum_sum2):
        bop_vec = []
        self.word_count = np.zeros((len(dataset), features_sax_size(self.alphabet_size)))
        for ts_i, ts in enumerate(dataset):
            cum_sum_ts = cum_sum[ts_i]
            cum_sum2_ts = cum_sum2[ts_i]
            self.bow = []
            self.transform_timeseries(ts_i, ts, cum_sum_ts, cum_sum2_ts)
            bop_vec.append(np.array(self.bow))

        return bop_vec, self.word_count

