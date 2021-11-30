from tqdm import tqdm
from dtaidistance import dtw
import avocado
import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)
from src.timeseries_object import TimeSeriesObject
import numpy as np
from tqdm.contrib.concurrent import process_map
from sklearn.metrics import balanced_accuracy_score

bands = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def _to_arr(dataset):
    res = []
    labels = []
    for ref in tqdm(dataset.objects, desc="loading data"):
        labels.append(ref.metadata["class"])
        res.append(TimeSeriesObject.from_astronomical_object(ref).to_fast_irregular_uts_object(bands))

    return np.array(res), np.array(labels)


class Classify:

    def __init__(self, train_name, test_name, train_max_chunks, test_max_chunks, n_jobs=8):
        self.train_name = train_name
        self.test_name = test_name
        self.train_max_chunks = train_max_chunks
        self.test_max_chunks = test_max_chunks
        self.test_values = None
        self.train_values = None
        self.train_labels = None
        self.n_jobs = n_jobs
        self.min_dists = None
        self.pred_lbl = None

    def classify(self):
        preds = None
        labels = None
        for chunk_test in tqdm(range(self.test_max_chunks), desc="Test dataset"):
            dataset_test = avocado.load(self.test_name, chunk=chunk_test, num_chunks=self.test_max_chunks)
            self.test_values, l_test  = _to_arr(dataset_test)
            self.min_dists = np.full(len(l_test), np.inf)
            self.pred_lbl = np.full(len(l_test), -1)
            for chunk_train in tqdm(range(self.train_max_chunks), desc="Train dataset"):
                dataset_train = avocado.load(self.train_name, chunk=chunk_train, num_chunks=self.train_max_chunks)
                self.train_values, self.train_labels = _to_arr(dataset_train)
                for i in tqdm(range(len(l_test)), desc="computing distances"):
                    for j in range(len(self.train_values)):
                        d = self.get_dist(self.test_values[i], self.train_values[j])
                        if d < self.min_dists[i]:
                            self.min_dists[i] = d
                            self.pred_lbl[i] = self.train_labels[j]

            if preds is None:
                preds = self.pred_lbl.copy()
                labels = l_test.copy()
            else:
                preds = np.append(preds, self.pred_lbl)
                labels = np.append(labels, l_test)

            break

        return preds, labels

    def get_dist(self, v1, v2):
        d = 0
        for b in bands:
            if v1[b] is not None and v2[b] is not None:
                d += dtw.distance_fast(v1[b].fluxes, v2[b].fluxes)
        return d

    def _classify(self, i):
        for j in range(len(self.train_values)):
            v1 = self.test_values[i]
            v2 = self.train_values[j]
            d = 0
            for b in bands:
                if v1[b] is not None and v2[b] is not None:
                    d += dtw.distance_fast(v1[b], v2[b])

            if d < self.min_dists[i]:
                self.min_dists[i] = d
                self.pred_lbl[i] = self.train_labels[j]
        return None


if __name__ == "__main__":

    train_name = "plasticc_train"
    test_name = "plasticc_test"
    n_chunks1 = 1
    n_chunks2 = 400
    clssfy = Classify(train_name, test_name, n_chunks1, n_chunks2)
    preds, labels = clssfy.classify()
    print("BALANCED ACC:", balanced_accuracy_score(preds, labels))