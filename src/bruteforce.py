from dtw import dtw, accelerated_dtw
import numpy as np
import multiprocessing as mp
import os

def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def twed(A, timeSA, B, timeSB, nu, _lambda):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP



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
                d, cost_matrix = twed(X_train[i], t_train[i], X_test[j], t_test[j], 1, 0.001)
                rows[i][j] = d
        out_q.put((i_subset, j_subset, rows))
    except:
        print("worker failed")
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
                d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(X_train[i], X_test[j], 'cosine')
                rows[i][j] = d
        out_q.put((i_subset, j_subset, rows))
    except:
        print("worker failed")
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
