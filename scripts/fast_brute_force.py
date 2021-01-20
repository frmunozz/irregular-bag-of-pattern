import numpy as np
from collections import defaultdict
import avocado
from tqdm import tqdm
from dtaidistance import dtw
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from src.timeseries_object import TimeSeriesObject
from tqdm.contrib.concurrent import process_map
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

bands = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]

def _to_arr(dataset, name):
    res = []
    labels = []
    for ref in tqdm(dataset.objects, desc="preparing dataset %s" % name):
        labels.append(ref.metadata["class"])
        res.append(TimeSeriesObject.from_astronomical_object(ref).to_fast_irregular_uts_object(bands))

    return np.array(res), np.array(labels)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="YlGnBu"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


class Classify:
    def __init__(self, train_name, test_name, n_jobs=6):
        self.d_train, self.l_train = _to_arr(avocado.load(train_name, verify_input_chunks=False), train_name)
        self.d_test, self.l_test = _to_arr(avocado.load(test_name, verify_input_chunks=False), test_name)
        self.n_jobs = n_jobs
        self._iter = range(len(self.l_test))

    def _dist(self, vi, vj):
        d = 0
        for b in bands:
            if vi[b] is not None and vj[b] is not None:
                d+= dtw.distance_fast(vi[b].fluxes, vj[b].fluxes)
        return d

    def _classify(self, i):
        vi = self.d_test[i]
        min_dist = np.inf
        pred = -1
        for j in range(len(self.l_train)):
            vj = self.d_train[j]
            d = self._dist(vi, vj)
            if d < min_dist:
                min_dist = d
                pred = self.l_train[j]
        return pred

    def classify(self):
        pred = process_map(self._classify, self._iter, max_workers=self.n_jobs, desc="classifying")
        print("balanced acc:", balanced_accuracy_score(self.l_test, pred))
        return pred

merged_labels = {
    6: 'Single microlens',
    15: 'TDE',
    16: 'Short period VS',
    42: 'SN',
    52: 'SN',
    53: 'Short period VS',
    62: 'SN',
    64: 'SN',
    65: 'M-dwarf',
    67: 'SN',
    88: 'AGN',
    90: 'SN',
    92: 'Short period VS',
    95: 'SN',
    99: 'Unknown',
}

plot_labels_extra_short = {
    6: 'Single $\mu$-lens',
    15: 'TDE',
    16: 'Eclip. Binary',
    42: 'SNII',
    52: 'SNIax',
    53: 'Mira',
    62: 'SNIbc',
    64: 'Kilonova',
    65: 'M-dwarf',
    67: 'SNIa-91bg',
    88: 'AGN',
    90: 'SNIa',
    92: 'RR lyrae',
    95: 'SLSN-I',
    99: 'Unknown',
}

if __name__ == '__main__':
    train_name = "plasticc_train_ddf_100"
    test_name = "plasticc_test_ddf_100"

    print("load data")
    clssfy = Classify(train_name, test_name)
    print("predict and clasify")
    pred = clssfy.classify()
    real = clssfy.l_test

    reorder = [6, 15, 65, 88, 42, 52, 62, 64, 67, 90, 95, 16, 53, 92]
    classes_names = [plot_labels_extra_short[x] for x in reorder]
    cnf_matrix2 = confusion_matrix(real, pred, labels=reorder)
    fig = plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix2, classes=classes_names, normalize=False,
                          title='Conf. matrix DTW-raw augment [b_acc:%.3f]' % balanced_accuracy_score(real, pred))
    plt.savefig(os.path.join(main_path, "data", "plasticc", "conf_matrix_dtw_raw_train_100.png",), dpi=300)

    # with train: 0.347 acc
    # with augment: 0.466 acc
