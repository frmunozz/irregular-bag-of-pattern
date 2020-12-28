from src.Adeprecated.core.timeseries import UTS, EmptyIrregularUTS, IrregularUTS
from src.Adeprecated.core import SlidingWindow
import numpy as np


class PiecewiseAggregateApproximation(object):
    def __init__(self, n, strategy="time_window", special_character=False):
        """Piecewise aggregate approximation

        transform a time series into a lower dimension version by applying PAA.
        The time series is segmented and each segments is represented by its mean value
        """
        self.n = n
        self.vec = np.zeros(n)
        self.strategy = strategy
        self.special_character = special_character

    def transform(self, sequence: IrregularUTS, big_window=None, ini=None):
        # pdb.set_trace()
        self.vec = np.full(self.n, np.NaN)
        if big_window is None:
            window = sequence.bandwidth() / self.n
        else:
            window = big_window / self.n

        sliding_window = SlidingWindow(window, _offset=1)
        sliding_window.initialize(sequence, ini=ini)
        i = 0
        while i < self.n:
            seq = sliding_window.get_sequence()
            if seq.size() == 0 and not self.special_character:
                return EmptyIrregularUTS()
            v = seq.mean()
            self.vec[i] = v
            i += 1
            if sliding_window.finish():
                break
            sliding_window.advance()

        if i == self.n or self.special_character:
            return UTS(self.vec)
        else:
            return EmptyIrregularUTS()
