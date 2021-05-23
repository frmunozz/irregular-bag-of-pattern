import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)

# from src.mmtva.tva import tva_distance_mp, classify_tva
from src.Adeprecated.mmtva.mtva import dataset_mtva_repr, generate_windows, mtva_distance_mp, classify_tva
from src.utils import load_numpy_dataset
import matplotlib.pyplot as plt
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

#	 print(cm)

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



if __name__ ==  '__main__':

	n_process = "default"

	data_path = os.path.join(main_path, "data", "plasticc_subsets", "ddf_wdf_split_min_detection", "3_min")
	output_report_path = os.path.join(data_path, "mtva_classification_reports")
	output_file_log = os.path.join(output_report_path, "tva_log.txt")
	output_file_results = os.path.join(output_report_path, "tva_results.csv")

	key = "wdf"

	train_base = key + "_train_%s.npy"
	test_base = key + "_test_%s.npy"
	text = "========INITIALIZE==========\n"
	text += "key: %s\n" % key
	text += "data path: %s\ntrain base: %s\ntest base: %s" % (data_path, train_base, test_base)
	print(text)
	f1_text = text + "\n"

	train_dataset, train_times, train_labels, n = load_numpy_dataset(data_path, train_base)
	test_dataset, test_times, test_labels, m = load_numpy_dataset(data_path, test_base)
	ts_sizes = [len(x) for x in train_dataset]
	ts_sizes.extend([len(x) for x in test_dataset])
	ts_width = [x[-1] - x[0] for x in train_times]
	ts_width.extend([x[-1] - x[0] for x in test_times])

	windows = generate_windows(ts_sizes, ts_width)
	windows = [28]
	print("using windows: ", windows)


	ini1 = time.time()
	train_mtva, train_sizes = dataset_mtva_repr(train_dataset, train_times, windows)
	test_mtva, test_sizes = dataset_mtva_repr(test_dataset, test_times, windows)
	end1 = time.time()
	text = "Time transform dataset to TVA form single-process: %s sec\n" % (str(end1-ini1))
	text += "mean size train set in tvia form: " + str(np.mean(train_sizes)) + "\n"
	text += "mean size test set in tvia form: " + str(np.mean(test_sizes))
	print(text)

	ini2 = time.time()
	result_pairs = mtva_distance_mp(train_mtva, test_mtva, len(windows), n_process=n_process)
	end2 = time.time()
	text = "Time to find closest between train and test TVA repr. multi-process: %s sec" % (str(end2-ini2))
	print(text)

	ini3 = time.time()
	real_labels, pred_labels, bacc, acc = classify_tva(result_pairs, train_labels, test_labels)
	end3 = time.time()
	text = "accuracy: %f, balanced_accuracy: %f\n" % (round(acc, 3), round(bacc, 3))
	text += "Time to classify: %s sec" % (str(end3-ini3))
	print(text)
