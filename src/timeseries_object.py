import pandas as pd
from avocado import AstronomicalObject
import numpy as np


class TimeSeriesObject:

    def __init__(self, observations: pd.DataFrame, **metadata_kwargs):
        self.observations = observations
        self.metadata = metadata_kwargs

    def __getitem__(self, item):
        if item == "mjd":
            item = "time"
        elif item == "passband":
            item = "band"
        return self.observations[item]

    def _single_band_sequence(self, band):
        return TimeSeriesObject(self.observations[self.observations["band"] == band], **self.metadata)

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
    def from_astronomical_object(cls, obj):
        metadata = obj.metadata
        observations = obj.observations

        return cls(observations, **metadata)