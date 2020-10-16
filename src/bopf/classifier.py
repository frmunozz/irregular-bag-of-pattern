import pandas as pd
from ..core.timeseries import IrregularUTS, UTS
import numpy as np
from collections import defaultdict
from scipy.stats import norm
import scipy.stats as stats
from .bop import features_sax_size


def _load_pandas(filename, passband_type="single", passband_id=3, **kwargs):
    df = pd.read_csv(filename + ".csv", **kwargs)
    df_metadata = pd.read_csv(filename + "_meta.csv", **kwargs)

    if passband_type == "multi":
        raise ValueError("multi-band support not implemented yet")

    df = df[df["passband"] == passband_id]
    df = df.sort_values(by=["object_id"])
    df_metadata = df_metadata.sort_values(by=["object_id"])
    df = df.groupby("object_id")
    fluxes = df['flux'].apply(list)
    times = df['mjd'].apply(list)
    ids = df.groups.keys()
    D = [IrregularUTS(fluxes.loc[i], times.loc[i]) for i in ids]
    labels = df_metadata["target"].to_numpy()

    return D, labels


def _load_file(filename, has_time=False, passband_type="single", passband_id=3, **kwargs):
    if has_time:
        raise ValueError("code not implemented for this case")

    if passband_type == "multi":
        raise ValueError("multi-band support not implemented yet")

    file1 = open(filename, 'r')
    lines = file1.readlines()
    D = []
    labels = []
    for d in lines:
        arr = d[:-1].split("\t")
        y = np.array(arr[1:], dtype=float)
        D.append(IrregularUTS(np.arange(y.size), y))
        labels.append(int(arr[0]))

    return D, labels


def cum_sum(dataset):
    cumsum = []
    cumsum2 = []

    for ts in dataset:
        x1_ts = np.zeros(ts.size())
        x2_ts = np.zeros(ts.size())
        for i, ts_y in enumerate(ts.y):
            x1_ts[i] = ts_y
            x2_ts[i] = ts_y ** 2
        cumsum.append(np.cumsum(x1_ts))
        cumsum2.append(np.cumsum(x2_ts))
    return cumsum, cumsum2


class Classifier(object):
    def __init__(self, train_dataset_path, test_dataset_path, fmt="pandas", discard_empty_segments=False, **kwargs):
        self.train, self.train_labels = self.load_dataset(train_dataset_path, fmt=fmt, **kwargs)
        self.test, self.test_labels = self.load_dataset(test_dataset_path, fmt=fmt, **kwargs)
        self.train_class_count = self.count_labels()
        self.labels_index = self.get_labels_index()
        self.cum_train, self.cum_train2 = cum_sum(self.train)
        self.cum_test, self.cum_test2 = cum_sum(self.test)
        self.alphabet_size = 4
        self.discard_empty_segments = discard_empty_segments

    def class_centroid(self, word_count_matrix, labels):
        centroids_sum = np.zeros((len(self.train_class_count), features_sax_size(self.alphabet_size)))
        for label, word_count in zip(labels, word_count_matrix):
            centroids_sum[self.labels_index[label]] += word_count
        for label, i in self.labels_index.items():
            centroids_sum[i] /= self.train_class_count[label]
        return centroids_sum

    def class_tf_idf(self):
        pass

    def classify(self):
        pass

    def ANOVA(self, word_count, labels_dataset):
        words_vec_size = word_count.shape[1]
        anova_f = np.zeros(words_vec_size)
        for j in range(words_vec_size):
            word_count_j = word_count[:, j]
            label_counts = defaultdict(list)
            is_not_zero = False
            for label_ts, count in zip(labels_dataset, word_count_j):
                is_not_zero = count > 0
                label_counts[label_ts].append(count)

            labels, values = zip(*label_counts.items())
            if is_not_zero:
                fvalue, pvalue = stats.f_oneway(*values)
            else:
                fvalue = 0
            anova_f[j] = fvalue
        return anova_f

    def count_labels(self):
        train_class_count = defaultdict(int)
        for l in self.train_labels:
            train_class_count[l] += 1
        return train_class_count

    def get_labels_index(self):
        return {label: i for i,label in enumerate(self.train_class_count)}

    def load_dataset(self, dataset_path, fmt="pandas", **kwargs):
        passband_type = kwargs.pop("passband_type", "single")
        passband_id = kwargs.pop("passband_id", 3)
        if fmt == "pandas":
            D, labels = _load_pandas(dataset_path, passband_type=passband_type,
                                     passband_id=passband_id, **kwargs)

        elif fmt == "file":
            has_time = kwargs.pop("has_time", False)
            D, labels = _load_file(dataset_path, passband_type=passband_type,
                                   passband_id=passband_id, has_time=has_time)
        else:
            raise ValueError("fmt unknown")

        class_count = defaultdict(int)
        for l in labels:
            class_count[labels] += 1
