from .timeseries import EmptyIrregularUTS, IrregularUTS
import pdb
import numpy as np

class SlidingWindow:
    def __init__(self, window, ini=0, _offset=0.5, join_sequences=False):
        """
        Sliding window object used for the extraction of sub-sequence of an UTS.

        :param window: window width, in JD.
        :param ini: initial time, in JD.
        :param _offset: displacement percent, value between 0 and 1.
        """
        self.window = window
        self.ini = ini
        self.offset = _offset
        self.uts = EmptyIrregularUTS()
        self.join_sequences = join_sequences
        self.i = 0
        self.j = 1

    def advance(self):
        """ advance window """
        delta = self.window * self.offset
        # if self.j > 1:
        #     prev_end_time = self.uts.t[self.j]
        # else:
        #     prev_end_time = self.uts.ini_time()
        if not self.finish():
            self.ini += delta
            # if prev_end_time < self.ini + delta and self.uts.t[self.j + 1] < prev_end_time + self.window:
            #     self.ini = prev_end_time
            # else:
            #     self.ini += delta

    def end(self):
        """ get end time of the window """
        return self.ini + self.window

    def initialize(self, uts, ini=None):
        """initialize the sliding window object for an specific UTS"""
        self.uts = uts
        if ini is None:
            ini = uts.ini_time()
        self.ini = ini
        self.i = 0
        self.j = 1

    def finish(self):
        """ check if the sub-sequence extraction process is finish"""
        return self.end() >= self.uts.end_time()

    def get_sequence(self):
        """ bet the sequence contained in the current position of the window """
        while self.i <= self.j < self.uts.size():
            cond_i = False
            cond_j = False
            if self.uts.t[self.i] < self.ini:
                self.i += 1
            else:
                cond_i = True
            if self.uts.t[self.j] <= self.end() and self.j + 1 < self.uts.size():
                self.j += 1
            else:
                cond_j = True

            if cond_i and cond_j:
                sequence = self.uts.get_sequence(self.i, self.j)
                return sequence
        return EmptyIrregularUTS()


class SlidingWindow2(object):
    def __init__(self, ts, window, advance_strategy="all_subsequences", tol=5):
        self.ts = ts
        self.window = window * self.ts.bandwidth()
        self.tol = tol
        self.i = 0
        self.j = 0
        self.end_time = ts.get_time(self.i)
        self.ini_time = ts.get_time(self.j)

    def _advance(self, idx):
        idx += 1
        return idx, self.ts.get_time(idx)

    def advance_j(self):
        self.j, self.end_time = self._advance(self.j)

    def advance_i(self):
        self.i, self.end_time = self._advance(self.i)

    def get_sequence(self):
        while self.j < self.ts.size():
            if self.end_time - self.ini_time < self.window:
                # advance j
                self.advance_j()
            else:
                if self.j - self.i - 1 > self.tol:
                    # return sequence range and advance i
                    seq_ini = self.i
                    seq_end = self.j
                    self.advance_i()
                    return seq_ini, seq_end
                else:
                    # advance i and j
                    self.advance_i()
                    self.advance_j()

        if self.j == self.ts.size():
            return -1, -1
        else:
            return self.i, self.j
