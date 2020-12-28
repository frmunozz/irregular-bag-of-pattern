import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
from collections import defaultdict
from src.Adeprecated.pydtw import IrregularDTW
import multiprocessing as mp
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import queue


def get_slope(times, fluxes):
	slope, intercept, r_value, p_value, std_err = stats.linregress(times,fluxes)
	return slope


def get_value(fluxes):
	return np.mean(fluxes)


def tva(times, fluxes, window):
	i = 0
	j = 1
	tva_vec = []
	ini_time = 0
	end_time = window
	while ini_time <  times[-1]:
		while times[i] < ini_time:
			i+=1
		while j < len(times) and times[j] < end_time:
			j+=1
		if j-i >= 1:
			time_seg = times[i:j]
			fluxes_seg = fluxes[i:j]
			if j-i > 1:
				slope = get_slope(time_seg, fluxes_seg)
			else:
				slope = 0
			val = get_value(fluxes_seg)
			tva_vec.append((True,(slope, val, i, j, ini_time, end_time)))
		else:
			tva_vec.append((False, (ini_time, end_time)))
		ini_time += window
		end_time += window
		i = j-1

	return tva_vec

def tva_to_arr(tva_data):
	arr = []
	for i in range(len(tva_data)):
		valid, values = tva_data[i]
		if valid:
			s, v, _, _, _, _ = values
			arr.append([s, v])
		else:
			arr.append(None)
	return arr



def plot_tva(tva_values, times, fluxes):
	plt.figure(figsize=(14, 4))
	means = []
	instants = []
	for k in range(len(tva_values)):
		not_null, values = tva_values[k]
		if not_null:
			s, v, i, j, ini_time, end_time = values
			j = j-1
			x = np.linspace(ini_time, end_time)
			v_time = ini_time + (end_time - ini_time) / 2
			new_intercep = v - v_time*s
			y = x * s + new_intercep
			means.append(v)
			instants.append(v_time)
			plt.plot(x, y, '-b', linewidth=2)
			plt.axvspan(ini_time, end_time, alpha=0.3, color='green')
		else:
			ini_time, end_time = values
			plt.axvspan(ini_time, end_time, alpha=0.1, color='black')
#	 x_min = (end_time - times1[i])/2
#	 y_min = x_min * s + intercep
#	 plt.plot([x_min], [y_min], 'ob', markersize=10)

	plt.plot(times, fluxes, '.-r', alpha=0.3)
	plt.plot(instants, means, 'ob', markersize=8)


def dataset_tva_repr(dataset_values, dataset_times, window=30):
	dataset_tva = []
	for i in range(len(dataset_values)):
		times = np.array(dataset_times[i])
		times = times - times[0]
		fluxes = preprocessing.scale(dataset_values[i])
		tva_repr = tva(times, fluxes, window)
		tva_arr = tva_to_arr(tva_repr)
		dataset_tva.append(tva_arr)

	return dataset_tva


def find_closest(train_tva, query_tva):
	min_dist = np.inf
	min_idx = -1
	for i in range(len(train_tva)):
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