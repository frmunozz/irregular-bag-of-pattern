from CTWED import ctwed
from dtaidistance import dtw 
import numpy as np
import multiprocessing as mp
import os


def distance_matrix(X_train, X_test):
    # compute distances between each time series on test set and train set
    n = len(X_train)
    m = len(X_test)
    res = np.ones((n, m)) * -1
    for i in range(n):
        for j in range(m):
            d = dtw.distance_fast(X_train[i].astype(float), X_test[j].astype(float), 'cosine')
            res[i][j] = d
    return res

def worker_twed(X_train, X_test, t_train, t_test, i_subset, j_subset, out_q):
    try:
        print("start worker twed: ", mp.current_process().name, "for [", i_subset, ", ", j_subset, "]")
        n = len(X_train)
        m = len(X_test)
        rows = np.ones((n, m)) * -1
        for i in range(n):
            if i % 10 == 0:
                print("worker twed '%s', i=%d/%d" % (mp.current_process().name, i, n))
            for j in range(m):
                d = ctwed(X_train[i].astype(float), t_train[i].astype(float), X_test[j].astype(float), t_test[j].astype(float), 0, 0.1)
                rows[i][j] = d
        out_q.put((i_subset, j_subset, rows))
    except Exception as e:
        print("worker failed with error code:", e)
    finally:
        print("worker '%s' done" % mp.current_process().name)


def worker(X_train, X_test, i_subset, j_subset, out_q):
    try:
        print("start worker dtw: ", mp.current_process().name, "for [", i_subset, ", ", j_subset, "]")
        n = len(X_train)
        m = len(X_test)
        rows = np.ones((n, m)) * -1
        for i in range(n):
            if i % 10 == 0:
                print("worker wtd '%s', i=%d/%d" % (mp.current_process().name, i, n))
            for j in range(m):
                d = dtw.distance_fast(X_train[i].astype(float), X_test[j].astype(float))
                rows[i][j] = d
        out_q.put((i_subset, j_subset, rows))
    except Exception as e:
        print("worker failed with error code:", e)
    finally:
        print("worker '%s' done" % mp.current_process().name)

def dmatrix_multiprocessing_v2(train_base, test_base, data_path, n_process=8, dist_type="dtw"):
    X_train = np.load(os.path.join(data_path, train_base % "d"), allow_pickle=True)
    X_test = np.load(os.path.join(data_path, test_base % "d"), allow_pickle=True)
    t_train = np.load(os.path.join(data_path, train_base % "t"), allow_pickle=True)
    t_test = np.load(os.path.join(data_path, test_base % "t"), allow_pickle=True)

    n = len(X_train)
    m = len(X_test)
    print("total shape to working: [%d,%d]" % (n, m))
    n_subset = 1 + n // n_process
    man = mp.Manager()
    result_queue = man.Queue()
    jobs = []

    for k in range(n_process):
        i_subset = k * n_subset
        j_subset = (k + 1) * n_subset
        if j_subset > n:
            j_subset = n
        X_train_subset = X_train[i_subset:j_subset]
        if dist_type == "dtw":
            jobs.append(mp.Process(target=worker,
                                   args=(X_train_subset, X_test, i_subset, j_subset, result_queue)))
        else:
            t_train_subset = t_train[i_subset:j_subset]
            jobs.append(mp.Process(target=worker_twed,
                                   args=(X_train_subset, X_test, t_train_subset, t_test, i_subset, j_subset, result_queue)))
        jobs[-1].start()

    for p in jobs:
        p.join()

    dmatrix = np.zeros((n, m))
    num_res = result_queue.qsize()
    while num_res > 0:
        i_subset, j_subset, rows_subset = result_queue.get()
        dmatrix[i_subset:j_subset] = rows_subset
        num_res -= 1

    return dmatrix


def dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder, dist_type="dtw"):
    X_train = np.load(in_folder + "train_d_n{}_c{}.npy".format(n1, c), allow_pickle=True)
    X_test = np.load(in_folder + "test_d_n{}_c{}.npy".format(n2, c), allow_pickle=True)
    t_train = np.load(in_folder + "train_t_n{}_c{}.npy".format(n1, c), allow_pickle=True)
    t_test = np.load(in_folder + "test_t_n{}_c{}.npy".format(n2, c), allow_pickle=True)
    n = len(X_train)
    m = len(X_test)
    n_subset = n // n_process
    man = mp.Manager()
    result_queue = man.Queue()
    jobs = []
    for k in range(n_process):
        i_subset = k * n_subset
        j_subset = (k + 1) * n_subset
        if j_subset > n:
            j_subset = n
        X_train_subset = X_train[i_subset:j_subset]
        if dist_type == "dtw":
            jobs.append(mp.Process(target=worker,
                                   args=(X_train_subset, X_test, i_subset, j_subset, result_queue)))
        else:
            t_train_subset = t_train[i_subset:j_subset]
            jobs.append(mp.Process(target=worker_twed,
                                   args=(X_train_subset, X_test, t_train_subset, t_test, i_subset, j_subset, result_queue)))
        print("Start process")
        jobs[-1].start()
        print("process started")

    for p in jobs:
        p.join()

    dmatrix = np.zeros((n, m))
    num_res = result_queue.qsize()
    while num_res > 0:
        i_subset, j_subset, rows_subset = result_queue.get()
        dmatrix[i_subset:j_subset] = rows_subset
        num_res -= 1

    if dist_type == "dtw":
        np.save(out_folder + "dmatrix_n{}_m{}_c{}.npy".format(n1, n2, c), dmatrix)
    else:
        np.save(out_folder + "dmatrix_n{}_m{}_c{}_twed.npy".format(n1, n2, c), dmatrix)

    return dmatrix
