from dtw import dtw, accelerated_dtw
import numpy as np
import multiprocessing as mp


def distance_matrix(X_train, X_test):
    # compute distances between each time series on test set and train set
    n = len(X_train)
    m = len(X_test)
    res = np.ones((n, m)) * -1
    for i in range(n):
        for j in range(m):
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(X_train[i], X_test[j], 'cosine')
            res[i][j] = d
    return res


def worker(X_train, X_test, i_subset, j_subset, out_q):
    try:
        n = len(X_train)
        m = len(X_test)
        rows = np.ones((n, m)) * -1
        for i in range(n):
            for j in range(m):
                d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(X_train[i], X_test[j], 'cosine')
                rows[i][j] = d
        out_q.put((i_subset, j_subset, rows))
    except:
        print("worker failed")
    finally:
        print("done")


def dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder):
    X_train = np.load(in_folder + "train_d_n{}_c{}.npy".format(n1, c), allow_pickle=True)
    X_test = np.load(in_folder + "test_d_n{}_c{}.npy".format(n2, c), allow_pickle=True)
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
        jobs.append(mp.Process(target=worker,
                               args=(X_train_subset, X_test, i_subset, j_subset, result_queue)))
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

    np.save(out_folder + "dmatrix_n{}_m{}_c{}.npy".format(n1, n2, c), dmatrix)

    return dmatrix
