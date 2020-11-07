import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)

from src.bopf.bopf_finder import bopf_param_finder_mp, bopf_best_classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd

import multiprocessing as mp
from collections import defaultdict
from src.bopf.bopf import BagOfPatternFeature
from src.bopf.classifier import classify, classify2
from src.utils import sort_trim_arr
import numpy as np
import time


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)


if __name__ ==  '__main__':
    wd_arr = [3, 4, 5, 6, 7]
    # step = 0.025
    # wl_arr = np.round((np.arange(int(1 / step)) + 1) * step, 3)
    # we discards windows smaller than 0.1
    # ini = 0.1
    # step = 0.025
    # wl_arr = ini + (np.arange(int((1 - ini) / step))) * step

    # wl_arr = np.array([50, 100, 125, 150, 175, 200, 225, 250, 300, 350])
    wl_arr = np.array([150, 175, 200])
    window_type = "time"
    n_process = 3
    top_n = 20

    # data_path = os.path.join(main_path, "data", "plasticc_subsets", "scenario1_ratio_2-8/")
    data_path = os.path.join(main_path, "data", "plasticc_subsets", "ddf_wdf_split_simplified", "60_40_split")
    output_report_path = os.path.join(data_path, "bopf_classification_reports")
    output_file_log = os.path.join(output_report_path, "bopf_finder_plasticc_log.txt")
    output_file_results = os.path.join(output_report_path, "bopf_finder_plasticc_results.csv")

    ddf_size_arr = [558, 930, 1860]
    wdf_size_arr = [1321, 2202, 4405]
    ddf_size_arr = ddf_size_arr[:1]
    wdf_size_arr = wdf_size_arr[:1]
    for size_arr, key in zip([wdf_size_arr], ["wdf"]):
        for size in size_arr:
            text1 = "========INITIALIZE==========\n"
            text2 = "key: %s\n" % key
            text3 = "dataset size: %d\n" % size
            train_base = key + "_train_%s_size" + str(size) + ".npy"
            test_base = key + "_test_%s_size" + str(size) + ".npy"
            text4 = "data path: %s\ntrain base: %s\ntest base: %s\n" % (data_path, train_base, test_base)
            print(text1, text2, text3, text4)
            ini1 = time.time()
            bopf, bopf_t, output_dict = bopf_param_finder_mp(data_path, train_base,
                                                             test_base, wd_arr, wl_arr,
                                                             n_process, window_type=window_type)
            end1 = time.time()
            ini2 = time.time()
            acc, s_index, pred_labels, real_labels, output_dict, _type = bopf_best_classifier(bopf, bopf_t,
                                                                                              output_dict, top_n,
                                                                                              window_type=window_type)
            end2 = time.time()
            text5 = "best accuracy of tests: %f\n" % acc
            text6 = "parameter finder execution time: %f\n" % (end1 - ini1)
            text7 = "test classifier execution time: %f\n" % (end2 - ini2)
            text8 = "=================================\n"
            print(text5, text6, text7, text8)
            cnf_matrix = confusion_matrix(real_labels, pred_labels)
            report = classification_report(real_labels, pred_labels, output_dict=True)
            df = pd.DataFrame(report)
            df.to_csv(os.path.join(output_report_path, "report_{}_size{}.csv".format(key, size)),
                      header=True, index=False)

            fig = plt.figure(figsize=(8, 8))
            plot_confusion_matrix(cnf_matrix, classes=bopf.tlabel, normalize=True,
                                  title='confusion matrix BOPF, dataset %s size %d' % (key, size))
            plt.savefig(os.path.join(output_report_path,
                                     "confusion_matrix_{}_size{}.png".format(key, size)),
                        dpi=300)

            f1 = open(output_file_results, "a")
            text_f1 = "{},{},{},{},{},{},{},{}\n".format(key, size, acc, output_dict["bop_wd"][s_index],
                                                            output_dict["bop_wl"][s_index], _type, end1 - ini1,
                                                            end2 - ini2)
            f1.write(text_f1)
            f1.close()

            f2 = open(output_file_log, "a")
            f2.write(text1 + text2 + text3 + text4 + text5 + text6 + text7 + text8)
            f2.close()