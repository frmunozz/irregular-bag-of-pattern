import pandas as pd
from avocado import AstronomicalObject
import numpy as np
from scipy.stats import linregress, norm
from scipy.interpolate import interp1d


class TimeSeriesObject:

    def __init__(self, observations: pd.DataFrame, **metadata_kwargs):
        self.observations = observations
        self.metadata = metadata_kwargs
        self.sorted = False

    def __getitem__(self, item):
        if item == "mjd":
            item = "time"
        elif item == "passband":
            item = "band"
        return self.observations[item]

    def sort_by(self, key):
        self.observations = self.observations.sort_values(key)
        self.sorted = True

    def _single_band_df(self, band):
        return self.observations[self.observations["band"] == band]

    def _single_band_value(self, band, key):
        return self._single_band_df(band)[key]

    def to_astronomical_object(self):
        return AstronomicalObject(self.metadata, self.observations)

    @property
    def n(self):
        return self.observations.shape[0]

    def flux_statistics(self):
        flux = self.observations["flux"].to_numpy(dtype=float)
        return np.mean(flux), np.std(flux)

    def to_numpy(self):
        return self.observations["mjd"], self.observations["flux"], self.observations["band"]

    @classmethod
    def from_astronomical_object(cls, obj, subtract_background=True):
        metadata = obj.metadata
        if subtract_background:
            observations = obj.subtract_background()
        else:
            observations = obj.observations

        return cls(observations, **metadata)

    def to_fast_irregular_uts_object(self, bands):
        if not self.sorted:
            self.observations = self.observations.sort_values("time")
            self.sorted = True
        df = self.observations.sort_values("time")
        g = df.groupby("band")
        fluxes = g["flux"].apply(np.hstack)
        times = g["time"].apply(np.hstack)
        objs = {}
        for b in bands:
            if b in fluxes:
                objs[b] = FastIrregularUTSObject(fluxes.loc[b], times.loc[b])
            else:
                objs[b] = None
        return objs

    def to_simple_object_for_numba(self, bands):
        if not self.sorted:
            self.observations = self.observations.sort_values("time")
            self.sorted = True
        df = self.observations.sort_values(["time"])
        g = df.groupby("band")
        fluxes = g["flux"].apply(np.hstack)
        times = g["time"].apply(np.hstack)
        v = {}
        t = {}
        for b in bands:
            if b in fluxes:
                v[b] = fluxes.loc[b]
                t[b] = times.loc[b]
            else:
                v[b] = np.array([])
                t[b] = np.array([])
        return v, t

    def fast_format_for_numba_code(self, bands_lbls):
        if not self.sorted:
            self.observations = self.observations.sort_values("time")
            self.sorted = True
        df = self.observations.sort_values(["band", "time"])
        bands = df["band"].to_numpy().astype(str)
        # print(len(bands))
        # print(bands)
        bands_ini = np.ones(len(bands_lbls)) * -1
        bands_end = np.ones(len(bands_lbls)) * -1
        for j, b in enumerate(bands_lbls):
            idxs = np.where(bands == b)[0]
            # print(idxs)
            if len(idxs) > 0:
                bands_ini[j] = int(np.min(idxs))
                bands_end[j] = int(np.max(idxs))

        # print(bands_ini, bands_end)
        fluxes = df["flux"].to_numpy().astype(float)
        times = df["time"].to_numpy().astype(float)
        return fluxes, times, bands_ini, bands_end


class FastIrregularUTSObject(object):
    def __init__(self, fluxes, times):
        self.fluxes = fluxes
        self.times = times
        self._n = len(fluxes)
        self.cum1, self.cum2 = self._cumsum()

    def _cumsum(self):
        cum1 = np.zeros(self._n + 1)
        cum2 = np.zeros(self._n + 1)
        psum = 0
        psum2 = 0
        for j in range(self._n):
            psum += self.fluxes[j]
            psum2 += self.fluxes[j] ** 2
            cum1[j + 1] = psum
            cum2[j + 1] = psum2
        return cum1, cum2

    def mean_sigma_segment(self, i, j):
        sumx = self.cum1[j] - self.cum1[i]
        sumx2 = self.cum2[j] - self.cum2[i]
        meanx = sumx / (j - i)
        meanx2 = sumx2 / (j - i)
        sigmax = np.sqrt(np.abs(meanx2 ** 2 - meanx ** 2))
        return meanx, sigmax

    def paa_value(self, i, j, meanx, sigmax):
        sumsub = self.cum1[j] - self.cum1[i]
        avgsub = sumsub / (j - i)
        if sigmax > 0:
            return (avgsub - meanx) / sigmax
        else:
            return 0

    def trend_value(self, i, j):
        t = self.times[i:j]
        y = self.fluxes[i:j]
        if len(t) < 2:
            return None
        if np.var(t) == 0:
            return np.pi/2
        if np.var(y) == 0:
            return 0
        slope, _, _, _, _ = linregress(t, y)
        return np.arctan(slope)

    def min_max_value(self, i, j):
        y = self.fluxes[i:j]
        if len(y) < 2:
            return None
        return np.max(y) - np.min(y)

    def std_value(self, i, j):
        y = self.fluxes[i:j]
        if len(y) < 2:
            return None
        return np.std(y)

    def mean_break_points(self, i, j, n, dist="normal"):
        vec = self.fluxes[i:j]
        if dist == "normal":
            # since we are using the cum sum technique it will always be
            # distributed with 0 mean and std 1
            return norm.ppf(np.linspace(0, 1, n + 1)[1:-1], 0, 1)
        else:
            return np.linspace(np.min(vec), np.max(vec), n+1)[1:-1]

    def min_max_break_points(self, i, j, n, **kwargs):
        vec = self.fluxes[i:j]
        diff = np.max(vec) - np.min(vec)
        return np.linspace(0, diff, n+1)[1:-1]

    def std_break_points(self, i, j, n, **kwargs):
        vec = self.fluxes[i:j]
        _max_std = np.std([np.min(vec), np.max(vec)])
        return np.linspace(0, _max_std, n+1)[1:-1]

    def trend_break_points(self, *args, **kwargs):
        return np.linspace(-np.pi / 2, np.pi / 2, args[2]+1)[1:-1]

    def interp1d(self, i, j):
        return interp1d(self.times[i:j], self.fluxes[i:j])

    def interp_paa_value(self, i, j):
        sumx = self.cum1[j] - self.cum1[i]
        sumx2 = self.cum2[j] - self.cum2[i]
        meanx = sumx / (j - i)
        meanx2 = sumx2 / (j - i)
        sigmax = np.sqrt(np.abs(meanx2 ** 2 - meanx ** 2))

        sumsub = self.cum1[j] - self.cum1[i]
        avgsub = sumsub / (j - i)
        if sigmax > 0:
            return (avgsub - meanx) / sigmax
        else:
            return 0

    def interp_trend_value(self, i, j):
        slope, _, _, _, _ = linregress(self.times[i:j], self.fluxes[i:j])
        return np.arctan(slope)

    def interp_min_max_value(self, i, j):
        return np.max(self.fluxes[i:j]) - np.min(self.fluxes[i:j])

    def interp_std_value(self, i, j):
        return np.std(self.fluxes[i:j])
