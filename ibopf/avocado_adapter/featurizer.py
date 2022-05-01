import avocado
import time
from ..timeseries_object import TimeSeriesObject


class AVOCADOFeaturizer(avocado.plasticc.PlasticcFeaturizer):

    def __init__(self, discard_metadata=False, record_times=False):
        self.discard_metadata = discard_metadata
        self.record_times = record_times
        self.records = []

    def select_features(self, raw_features):

        features = super(AVOCADOFeaturizer, self).select_features(raw_features)

        if self.discard_metadata:
            features.pop("host_photoz")
            features.pop("host_photoz_error")

        return features

    def extract_raw_features(self, astronomical_object, return_model=False):
        ini = time.time()
        raw_features = super(AVOCADOFeaturizer, self).extract_raw_features(astronomical_object, return_model=return_model)
        end = time.time()

        if self.record_times:
            self.records.append([len(astronomical_object.observations), end - ini])

        return raw_features


class MMMBOPFFeaturizer(avocado.plasticc.PlasticcFeaturizer):

    def __init__(self, include_metadata=False, metadata_keys=None, method=None, zero_variance_model=None, compact_model=None):
        if metadata_keys is None:
            metadata_keys = ["host_photoz", "host_photoz_error"]

        self.metadata_keys = metadata_keys
        self.include_metadata = include_metadata
        self.metadata = None
        self.method = method
        self.zero_variance_model = zero_variance_model
        self.compact_model = compact_model

    def select_features(self, raw_features):
        # in this case raw features are the compact features
        # and we are going to append the metadata features
        if self.include_metadata:
            for k in self.metadata_keys:
                raw_features.loc[:, k] = self.metadata[k]

        return raw_features

    def extract_raw_features(self, astronomical_object, return_model=False):
        if self.method is None:
            raise ValueError("cannot run extraction without the method")

        data = TimeSeriesObject.from_astronomical_object(astronomical_object).fast_format_for_numba_code(astronomical_object.bands)

        sparse_data = self.method.mmm_bopf(data)

        if self.zero_variance_model is not None:
            sparse_data = self.zero_variance_model.transform(sparse_data)

        if self.compact_model is not None:
            compact_data = self.compact_model.transform(sparse_data)
            return compact_data
        else:
            return sparse_data