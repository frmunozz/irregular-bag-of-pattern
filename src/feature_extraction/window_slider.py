import numpy as np
from src.timeseries_object import FastIrregularUTSObject


class Slider(object):
    def __init__(self, window, step=None, index_based=False, overlapping=True):
        """
        Slider object for sub-sequence extraction from time series dataset.

        For each time series, a fixed window is slided across the time series,
        extracting the corresponding sub-sequences indices (start and end point).

        :param window: the window width
        :param step: (optional) a defined fixed step. By default the method use
                    a variable step based on element index.
        :param index_based: (optional) If True, apply the slider using the indexes.
                            Otherwise, generate an splider window based on value array
                            which has to be sorted in increasing order.
        :param overlapping: (optional) If false, the generated set of sub-sequence will be disjoint.
        """
        self.window = window
        self.step = step
        self.index_based = index_based
        self.overlapping = overlapping

    def _get_starts(self, first_ini, last_end, step):
        n = int((last_end - first_ini) / step)
        return np.arange(n) * step + first_ini

    def _define_ranges(self, values: np.ndarray):
        first_ini = values[0]
        last_end = values[-1]

        if self.step is None:
            if self.overlapping:
                step = None
            else:
                step = self.window
        else:
            if self.overlapping:
                step = self.step + self.window
            else:
                step = self.step

        if step is None:
            window_starts = values[:-1]
        else:
            window_starts = self._get_starts(first_ini, last_end, step)

        window_ends = window_starts + self.window
        return window_starts, window_ends

    def _index_based_extract(self, values: np.ndarray, offset_idx=0):
        assert isinstance(self.window, int)
        if self.step is None:
            step = 1
        else:
            step = self.step

        if not self.overlapping:
            step = self.window
        n = values.size
        initials = np.arange(n // step) * step + offset_idx
        ends = initials + self.window + offset_idx
        return np.array((initials, ends)).T

    def _value_based_extract(self, values: np.ndarray, offset_idx=0):
        initials, ends = self._define_ranges(values)
        i = 0
        j = 0
        n = len(initials)
        windows = np.zeros((n, 2), dtype=int) - 1
        for k in range(n):
            while i < n and values[i] < initials[k]:
                i += 1
            while j < n and values[j] <= ends[k]:
                j += 1
            if i >= j:
                windows[k] = -1, -1
            else:
                windows[k] = [i + offset_idx, j + offset_idx]
        return windows

    def extract_from_multi_band_ts(self, mb_data: dict):

        # get_by_bands
        res_dict = {}
        for k, v in mb_data.items():
            if self.index_based:
                windows = self._index_based_extract(v)
            else:
                windows = self._value_based_extract(v)
            res_dict[k] = windows

        return res_dict

    def extract_from_vector(self, vec, offset_idx=0):
        if self.index_based:
            segments = self._index_based_extract(vec, offset_idx=offset_idx)
        else:
            segments = self._value_based_extract(vec, offset_idx=offset_idx)
        return segments


class TwoWaysSlider(object):
    def __init__(self, window, tol=5):
        self._window = window
        self._tol = tol
        self._forward_sequences = []
        self._backward_sequences = []
        self._n0 = 0

    def _get_starts(self, first_ini, last_end, step):
        n = int((last_end - first_ini) / step)
        return np.arange(n) * step + first_ini

    def _forward_slide(self, values: np.ndarray):
        starts = values[:-1]
        self._forward_sequences = []
        j = 0
        n = values.size
        for i, ini in enumerate(starts):
            while j < n and values[j] <= ini + self._window:
                j += 1
            if j - i > self._tol:
                self._forward_sequences.append([i, j, ini])

    def _backward_slide(self, values: np.ndarray):
        ends = values[1:]
        self._backward_sequences = []
        n = values.size
        i = n-1
        for j, end in enumerate(ends):
            while i > 0 and values[i] > end - self._window:
                i -= 1
            i += 1  # correct bound to include index i in window
            if j - i > self._tol:
                self._backward_sequences.append([i, j, end])

    def fit(self, values: np.ndarray):
        self._forward_slide(values)
        self._backward_slide(values)
        self._n0 = len(self._forward_sequences)

    def n(self):
        return self._n0

    def get_subsequence(self, k):
        if k >= self._n0:
            i, j, end = self._backward_sequences[k]
            ini = end - self._window
        else:
            i, j, ini = self._forward_sequences[k]
            end = ini + self._window
        return i, j, ini, end


class Segmentator:

    def __init__(self, window, k, index_based=False):
        self.window = window
        self.k = k
        self.sub_win = self.window / self.k
        self.index_based = index_based

    def _value_based_segmentate(self, values: np.ndarray, offset, ini_time):
        i = 0
        j = 1
        n = len(values)
        segments = []
        empty_segments = []
        end_time = ini_time + self.sub_win
        for k in range(self.k):
            while i < n-1 and values[i] < ini_time:
                i += 1
            while j < n and values[j] <= end_time:
                j += 1
            # print(i, j)
            if i == n or values[i] >= end_time or values[j-1] < ini_time:
                segments.append([-1, -1])
                empty_segments.append(k)
            else:
                segments.append([i + offset, j + offset])
            ini_time += self.sub_win
            end_time += self.sub_win
            i = j
        return np.array(segments), np.array(empty_segments)

    def _index_based_segmentate(self, values: np.ndarray, offset, ini_time):
        n = len(values)
        sub_win = n // self.k
        segments = []
        for k in range(self.k):
            segments.append([k * sub_win + offset, (k+1) * sub_win + offset])

        return segments, np.array([])

    def segmentate(self, ts: FastIrregularUTSObject, i, j, ini_time):
        if self.index_based:
            return self._index_based_segmentate(ts.times[i:j], i, ini_time)
        else:
            return self._value_based_segmentate(ts.times[i:j], i, ini_time)
