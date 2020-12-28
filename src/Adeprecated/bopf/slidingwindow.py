import pdb


class SlidingWindow(object):
    def __init__(self, ts, window, advance_strategy="all_subsequences", tol=5):
        self.ts = ts
        self.window = window * self.ts.bandwidth()
        self.tol = tol
        self.i = 0
        self.j = 1
        self.end_time = ts.get_time(self.i)
        self.ini_time = ts.get_time(self.j)

    def _advance(self, idx):
        idx += 1
        return idx, self.ts.get_time(idx)

    def advance_j(self):
        self.j, self.end_time = self._advance(self.j)

    def advance_i(self):
        self.i, self.ini_time = self._advance(self.i)

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

        if self.j == self.ts.size() or self.j - self.i - 1 < self.tol:
            return -1, -1
        else:
            return self.i, self.j
