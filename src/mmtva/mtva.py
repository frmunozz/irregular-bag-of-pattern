import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict
from ..pydtw import IrregularDTW
from .tva import tva
import multiprocessing as mp
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import queue


def mtva_arr(times, fluxes, windows):
	vec = []
	size = 0
	for window in windows:
		tva_repr = tva(times, fluxes, window)
		tva_repr_arr = _mtva_to_arr(tva_repr)
		vec.append(tva_repr_arr)
		size += len(tva_repr_arr)
	return vec, size

def _mtva_to_arr(tva_data):
	arr = []
	for i in range(len(tva_data)):
		valid, values = tva_data[i]
		if valid:
			s, v, _, _, _, _ = values
			arr.append([s, v])
		# else:
			# for uni band its discard when its none.
			# for multi-band its discard when all bands are none.
			# arr.append(None)
	return arr

def generate_windows(ts_sizes, ts_width):
	mean_size = np.mean(ts_sizes)
	mean_width = np.mean(ts_width)
	l_max = int(np.log2(mean_size/2)) + 1
	windows = []
	for i in range(l_max):
		windows.append(np.round(mean_width / (2**(l_max - 1 - i)), 3))
	print("Datset of mean size:", round(mean_size, 3), " and mean width:", round(mean_width, 3))
	return windows

def dataset_mtva_repr(dataset_values, dataset_times, windows):
	dataset_mtva = []
	mtva_sizes = []
	for i in range(len(dataset_values)):
		times = np.array(dataset_times[i])
		times = times - times[0]
		fluxes = preprocessing.scale(dataset_values[i])
		mtva_repr_arr, size = mtva_arr(times, fluxes, windows)
		dataset_mtva.append(mtva_repr_arr)
		mtva_sizes.append(size)

	return dataset_mtva, mtva_sizes


def find_closest(train_tva, query_tva, n):
	min_dist = np.inf
	min_idx = -1
	for i in range(len(train_tva)):
		d = 0
		for j in range(n):
			idtw = IrregularDTW(train_tva[i][j], query_tva[j])
			idtw.set_matrix()
			idtw.compute()
			d += idtw.distance()
		if d < min_dist:
			min_dist = d
			min_idx = i

	return min_dist, min_idx




def mtva_distance_worker(tvas_to_query, train_tva, n, lock, out_q):

	try: 
		print("start worker '%s'" % mp.current_process().name)

		output_dict = defaultdict(list)
		c = 0
		while True:
			try:
				lock.acquire()
				query_tva, query_idx = tvas_to_query.get_nowait()
			except queue.Empty:
				lock.release()
				break
			else:
				lock.release()
				min_dist, min_idx = find_closest(train_tva, query_tva, n)
				c += 1
				out_q.put((query_idx, min_idx, min_dist))
				if c % 10 == 1:
					print("worker '%s' has processed %d queries" % (mp.current_process().name, c))


	except Exception as e:
		print("Worker failed with error:", e)
	finally:
		print("worker '%s' DONE" % mp.current_process().name)

def mtva_distance_mp(train_tva, test_tva, n, n_process="default"):
	if n_process == "default":
		n_process = mp.cpu_count()

	m = mp.Manager()
	result_queue = m.Queue()


	lock = mp.Lock()

	tvas_to_query = mp.Queue()

	for i in range(len(test_tva)):
		tvas_to_query.put((test_tva[i], i))

	lock = mp.Lock()

	print("total queries to process: %d" % len(test_tva))

	jobs = []
	for w in range(n_process):
		p = mp.Process(target=mtva_distance_worker, args=(tvas_to_query, train_tva, n, lock, result_queue))
		jobs.append(p)
		p.start()

	for p in jobs:
		p.join()


	num_res = result_queue.qsize()
	result_pairs = []
	while num_res > 0:
		query_idx, closest_idx, d = result_queue.get()
		result_pairs.append([query_idx, closest_idx, d])
		num_res -=1

	return result_pairs

def classify_tva(result_pairs, train_label, test_label):
	real_labels = []
	pred_labels = []
	for pair in result_pairs:
		test_idx, train_idx, distance = pair
		real_labels.append(test_label[test_idx])
		pred_labels.append(train_label[train_idx])

	bacc = balanced_accuracy_score(real_labels, pred_labels)
	acc = accuracy_score(real_labels, pred_labels)
	return real_labels, pred_labels, bacc, acc