from .context import src
import unittest
import numpy as np


def gen_timeseries(n):
    t = np.arange(n) * 1.2
    Fs = 80
    f = 5
    y = np.sin(2 * np.pi * f * t / Fs)
    return src.IrregularUTS(t, y, metadata={"target": 1})


class TestSlidingWindow(unittest.TestCase):

    def test_initialize(self):
        n = 100
        ts = gen_timeseries(100)
        ini = ts.t[n//3]
        window = ts.t[n//4]
        swindow = src.SlidingWindow(window)
        swindow.initialize(ts, ini=ini)
        self.assertEqual(swindow.ini, ini)
        self.assertEqual(swindow.uts.y.shape, ts.y.shape)
        self.assertEqual(swindow.uts.t.shape, ts.t.shape)
        self.assertEqual(swindow.i, 0)
        self.assertEqual(swindow.j, 1)

    def test_advance(self):
        n = 100
        ts = gen_timeseries(100)
        ini = ts.t[n // 3]
        window = ts.t[n // 4]
        swindow = src.SlidingWindow(window)
        self.assertEqual(swindow.ini, 0)
        swindow.advance()
        self.assertEqual(swindow.ini, 0)
        swindow.initialize(ts)
        swindow.advance()
        self.assertEqual(swindow.ini, window * 0.5)

    def test_get_sequence(self):
        n = 100
        ts = gen_timeseries(100)
        ini = ts.t[n // 3]
        window = ts.t[n // 4]
        swindow = src.SlidingWindow(window)
        swindow.initialize(ts, ini=ini)
        seq = swindow.get_sequence()
        self.assertTrue(ini <= seq.ini_time())
        self.assertTrue(swindow.end() >= seq.end_time())
        self.assertTrue(window >= seq.bandwidth())


class TestAlphabet(unittest.TestCase):

    def test_word_to_number(self):
        pass


class TestPAA(unittest.TestCase):
    pass


class TestSAX(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
