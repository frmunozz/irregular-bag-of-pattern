import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)

from src.bruteforce import dmatrix_multiprocessing_v2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import itertools
import pandas as pd


def bruteforce_classifier(data_path, train_base, test_base, dmatrix):

	train_labels = np.load(os.path.join(data_path, train_base % "l"), allow_pickle=True)
	test_labels = np.load(os.path.join(data_path, test_base % "l"), allow_pickle=True)

	n = len(train_labels)
	m = len(test_labels)
	pred_labels = []
	for j in range(m):
		dmin = np.inf
		mink = -1
		for i in range(n):
			if dmatrix[i][j] < dmin:
				dmin = dmatrix[i][j]
				mink = i

		pred_labels.append(train_labels[mink])

	balanced_acc = balanced_accuracy_score(test_labels, pred_labels)
	acc = accuracy_score(test_labels, pred_labels)

	return test_labels, pred_labels, acc, balanced_acc


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

if __name__ == "__main__":

	ddf_size_arr = [558, 930, 1860]
	wdf_size_arr = [1321, 2202, 4405]
	n_process = 8

	data_path = os.path.join(main_path, "data", "plasticc_subsets", "ddf_wdf_split_simplified", "60_40_split")
	output_report_path = os.path.join(data_path, "brute_force_classification_reports")
	output_file_log = os.path.join(output_report_path, "dmatrix_plasticc_log.txt")

	for dist_type in ["dtw", "twed"]:
		for size_arr, key in zip([wdf_size_arr[:2]], ["wdf"]):
			for size in size_arr:
				train_base = key + "_train_%s_size" + str(size) + ".npy"
				test_base = key + "_test_%s_size" + str(size) + ".npy"
				ini = time.time()
				dmatrix = dmatrix_multiprocessing_v2(train_base, test_base, data_path, n_process=n_process, dist_type=dist_type)
				end = time.time()

				ini2 = time.time()
				real, pred, acc, balanced_acc = bruteforce_classifier(data_path, train_base, test_base, dmatrix)
				end2 = time.time()

				cnf_matrix = confusion_matrix(real, pred)
				report = classification_report(real, pred, output_dict=True)
				df = pd.DataFrame(report)
				df.to_csv(os.path.join(output_report_path, "report_{}_{}_size{}.csv".format(key, dist_type, size)),
					header=True, index=False)
				fig = plt.figure(figsize=(8, 8))
				plot_confusion_matrix(cnf_matrix, classes=np.unique(real), normalize=True,
					title='confusion matrix BF-%s, dataset %s size %d' % (dist_type, key, size))
				plt.savefig(os.path.join(output_report_path,
					"confusion_matrix_{}_{}_size{}.png".format(key, dist_type, size)), dpi=300)

				np.save(os.path.join(output_report_path, key + "_dmatrix_" + dist_type + "_size" + str(size) + ".npy"), dmatrix)
				text = "===================================\n"
				text += "key: %s, size: %d, dist type: %s" % (key, size, dist_type)
				text += "dmatrix shape: (%d, %d)\n" % (dmatrix.shape[0], dmatrix.shape[1])
				text +=  "dmatric computing time (sec): " + str(round(end-ini, 3)) + "\n"
				text += "classification time (sec): " + str(round(end2-ini2, 3)) + "\n"
				text += "acc: " + str(round(acc, 3)) + ", balanced acc: " + str(round(balanced_acc, 3)) + "\n"
				text += "===================================\n"
				print(text)
				f = open(output_file_log, "a")
				f.write(text)
				f.close()
