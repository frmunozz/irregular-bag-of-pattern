import os
import sys
import numpy as np

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)

from src.preprocesing import gen_dataset_from_h5
from src.mb_bruteforce import mb_dmatrix_mp
import time
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

band_map = {
    0: 'lsstu',
    1: 'lsstg',
    2: 'lsstr',
    3: 'lssti',
    4: 'lsstz',
    5: 'lssty',
}


def acc_loo(dmatrix, labels):
    n = dmatrix.shape[0]
    pred = []
    for i in range(n):
        row = dmatrix[i]
        idxs = np.argsort(row)
        if idxs[0] != i:
            raise ValueError("we have a problem")
        pred.append(labels[idxs[1]])
    return pred, balanced_accuracy_score(labels, pred)


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


def conf_matrix_reduce_test_labels(cm_full, n_reduced, labels_order, labels_transf, labels_transf2):
    cm_reduced = np.zeros((n_reduced, cm_full.shape[1]))
    for i in range(cm_full.shape[0]):
        for j in range(cm_full.shape[1]):
            k = labels_transf2[labels_transf[labels_order[i]]]
            cm_reduced[k, j] += cm_full[i, j]
    return cm_reduced


merged_labels_to_num = {
    'Single microlens': 1,
    'TDE': 2,
    'Short period VS': 3,
    'SN': 4,
    'M-dwarf': 5,
    'AGN': 6,
    'Unknown': 99
}

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

plot_merged_labels = {
    1: 'Single microlens',
    2: 'TDE',
    3: 'Short period VS',
    4: 'SN',
    5: 'M-dwarf',
    6: 'AGN',
}

if __name__ == "__main__":
    res, labels, metadata = gen_dataset_from_h5("plasticc_balanced_full_wfd")
    # res, labels, metadata2 = gen_dataset_from_h5("plasticc_balanced_combined_classes_small_wfd")

    # combination of classes
    # labels = [merged_labels_to_num[merged_labels[x]] for x in labels]
    # labels2 = [merged_labels_to_num[merged_labels[x]] for x in labels2]

    # print(np.unique(labels), np.unique(labels2))
    for i in tqdm(
            range(len(res)), desc="Preprocessing data", dynamic_ncols=True
    ):
        obs1 = res[i].observations
        obs1 = obs1.sort_values(by=["time"])
        obs_arr = np.zeros(6, dtype=object)
        for j, b in band_map.items():
            tmp1 = obs1[obs1["band"] == b]
            obs_arr[j] = tmp1["flux"].to_numpy(dtype=float)

        res[i] = obs_arr

    print("preprocessed")
    # res = res[:100]
    ini = time.time()
    dmatrix = mb_dmatrix_mp(res, n_process=6)
    end = time.time()
    print("comutation process complete, time: ", end - ini)
    np.save(os.path.join(main_path, "data", "plasticc", "balanced_full_wfd_{}_dmatrix".format(len(res))), dmatrix)
    print("dmatrix saved")
    print("END")

    # print("getting wfd dmatrix")
    # dmatrix = np.load(os.path.join(
    #     main_path, "data", "plasticc", "balanced_combined_wfd_{}_dmatrix.npy".format(len(res))),
    #     allow_pickle=True)
    # print("getting ddf dmatrix")
    # dmatrix = np.load(
    #     os.path.join(main_path, "data", "plasticc", "balanced_full_ddf_{}_dmatrix.npy".format(len(res))),
    #     allow_pickle=True)

    # print("accuracy leave-one-out wfd")
    # pred2, acc2 = acc_loo(dmatrix, labels)
    # print("accuracy: ", acc2)
    # fig2 = plt.figure(figsize=(8, 8))
    # cnf_matrix2 = confusion_matrix(labels, pred2)
    # plot_confusion_matrix(cnf_matrix2, classes=np.unique(labels).astype(int), normalize=True,
    #                       title='confusion matrix BF-%s, dataset %s' % ("DTW", "WFD"))
    # plt.savefig(os.path.join(main_path, "data", "plasticc",
    #                          "conf_matrix_balanced_combined_wfd_{}_bf_dtw.png".format(len(res))), dpi=300)

    # print("accuracy leave-one-out ddf")
    # pred, acc = acc_loo(dmatrix, labels)
    # print("accuracy: ", acc)
    # fig = plt.figure(figsize=(12, 12))
    # full_labels_order = [6, 15, 65, 88, 42, 52, 62, 64, 67, 90, 95, 16, 53, 92]
    # full_labels_order = [1, 2, 5, 6, 3, 4]
    # label_names = np.array([plot_labels_extra_short[x] for x in full_labels_order])
    # label_names = np.array([plot_merged_labels[x] for x in full_labels_order])
    # cnf_matrix = confusion_matrix(labels, pred, labels=full_labels_order)
    # print("conf m full shape", cnf_matrix.shape)
    # plot_confusion_matrix(cnf_matrix, classes=label_names, normalize=False,
    #                       title='confusion matrix BF-%s, dataset %s' % ("DTW", "DDF"))
    # plt.savefig(os.path.join(main_path, "data", "plasticc",
    #                          "conf_matrix_balanced_full_merged_ddf_{}_bf_dtw.png".format(len(res))), dpi=300)

    # labels_transf = {
    #     6: 1,
    #     15: 2,
    #     16: 3,
    #     42: 4,
    #     52: 4,
    #     53: 3,
    #     62: 4,
    #     64: 4,
    #     65: 5,
    #     67: 4,
    #     88: 6,
    #     90: 4,
    #     92: 3,
    #     95: 4,
    #     99: 99,
    # }
    #
    # labels_transf2 = {
    #     1: 0,
    #     2: 1,
    #     3: 5,
    #     4: 4,
    #     5: 2,
    #     6: 3
    # }
    #
    # labels_transf2_inv = {
    #     0: 1,
    #     1: 2,
    #     5: 3,
    #     4: 4,
    #     2: 5,
    #     3: 6
    # }
    # reduced_labels_order = [labels_transf2[y] for y in np.unique([labels_transf[x] for x in full_labels_order])]
    # print(reduced_labels_order)
    # cnf_red = conf_matrix_reduce_test_labels(cnf_matrix, 6, full_labels_order, labels_transf, labels_transf2)
    # print("conf m reduced shape", cnf_red.shape)
    # aylabels = ['Single $\mu$-lens', 'TDE', 'M-dwarf', 'AGN', 'SN', 'Short period VS']
    # axlabels = [plot_labels_extra_short[x] for x in full_labels_order]
    # plt.figure(figsize=(12, 6))
    # sn.set(font_scale=1.4)  # for label size
    # g = sn.heatmap(cnf_red, cmap="YlGnBu", annot=True, fmt="g", annot_kws={"size": 12}, xticklabels=axlabels, yticklabels=aylabels)  # font size
    # g.set_xticklabels(g.get_xticklabels(), rotation=60, ha="right")
    # plt.xlabel("Predict full class")
    # plt.ylabel("Real merged class")
    # plt.title("Comparison between full classes and merge classes, DDF full")
    # plt.tight_layout()
    # plt.show()
