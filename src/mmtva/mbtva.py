import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict
from ..pydtw import IrregularDTW
import multiprocessing as mp
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import queue


def get_slope(times, fluxes):
	slope, intercept, r_value, p_value, std_err = stats.linregress(times,fluxes)
	return slope


def get_value(fluxes):
	return np.mean(fluxes)


def mb_tva(times_mb, fluxes_mb, window, n_bands=6):
	i = 0
	j = 1
	mb_tva_vec = []
	mb_tva_arr = []
	ini_time = 0
	end_time = window
	widths = np.array([x[-1] - x[0] for x in times_mb])
	i_mb = np.zeros(n_bands)
	j_mb = np.zeros(n_bands)
	idx = np.argmax(widths)
	while ini_time <  times_mb[idxs][-1]:

		mb_vec = np.full((n_bands, 3), np.nan)

		atleast_1_valid = False
		for b in range(n_bands):
			i = i_mb[b]
			j = j_mb[b]
			while times[b][i] < ini_time:
				i += 1

			j = i
			while j < len(times[b]) and times[b][j] <= end_time:
				j += 1

			if j-i >= 1:
				atleast_1_valid = True
				time_seg = times[b][i:j]
				fluxes_seg = fluxes[b][i:j]
				if j-i > 1:
					slope = get_slope(time_seg, fluxes_seg)
				else:
					slope = 0
				val = get_value(fluxes_seg)
				mb_vec[b] = [slope, val, ini_time]
				# tva_vec.append((True,(slope, val, i, j, ini_time, end_time)))
			# else:
				# tva_vec.append((False, (ini_time, end_time)))
			i_mb[b] = j-1
			j_mb[b] = j

		if atleast_1_valid:
			mb_tva_arr.append(mb_vec[:,:2])

		ini_time += window
		end_time += window
		i = j-1

	return np.array(mb_tva_arr)


def dataset_mb_tva_repr(dataset_values_mb, dataset_times_mb, window=30, n_bands=6):
	dataset_mb_tva = []
	for i in range(len(dataset_values_mb)):
		times_mb = dataset_times_mb[i]
		min_time = min([x[0] for x in times_mb])
		fluxes_mb = preprocessing.scale(dataset_values[i])

		for b in range(n_bands):
			times_mb[b] = times_mb[b] - min_time
			fluxes_mb[b] = preprocessing.scale(fluxes_mb[b])
		mb_tva_arr = mb_tva(times_mb, fluxes_mb, window, n_bands=n_bands)
		dataset_mb_tva.append(mb_tva_arr)

	return dataset_mb_tva


def find_closest(train_mb_tva, query_mb_tva):
	min_dist = np.inf
	min_idx = -1
	for i in range(len(train_mb_tva)):
		idtw = IrregularDTW(train_tva[i], query_tva)
		idtw.set_matrix()
		idtw.compute()
		if idtw.distance() < min_dist:
			min_dist = idtw.distance()
			min_idx = i

	return min_dist, min_idx




def tva_distance_worker(tvas_to_query, train_tva, lock, out_q):

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
				min_dist, min_idx = find_closest(train_tva, query_tva)
				c += 1
				out_q.put((query_idx, min_idx, min_dist))
				if c % 10 == 1:
					print("worker '%s' has processed %d queries" % (mp.current_process().name, c))


	except Exception as e:
		print("Worker failed with error:", e)
	finally:
		print("worker '%s' DONE" % mp.current_process().name)

def tva_distance_mp(train_tva, test_tva, n_process="default"):
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
		p = mp.Process(target=tva_distance_worker, args=(tvas_to_query, train_tva, lock, result_queue))
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