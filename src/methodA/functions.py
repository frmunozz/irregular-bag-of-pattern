import multiprocessing as mp
import queue
from .transformer2 import BOPTransformer
from .representation import BOPSparseRepresentation
from .reducer import PCAReducer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.decomposition import PCA
from ..predictors import predict_by_cosine, predict_by_euclidean
from sklearn.metrics import balanced_accuracy_score
import time


def _transformer_worker(dataset: np.ndarray, lock: mp.synchronize.Lock,
                        combinations_to_try: mp.queues.Queue,
                        result_queue: mp.queues.Queue, **kwargs):
    try:
        while True:
            try:
                lock.acquire()
                i, win, wl = combinations_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                print("'{}' processing params [win {}, wl {}]".format(mp.current_process().name,
                                                                      win, wl))
                transf = BOPTransformer(window=win, word_length=wl,
                                        **kwargs)
                bop_repr, failed = transf.transform_dataset(dataset)
                result_queue.put((i, bop_repr, failed))
    except Exception as e:
        print("worker '%s' failed with error: " % mp.current_process().name, e)


def transform_to_bop(dataset: np.ndarray, tuples, multiprocessing=True, **kwargs):
    n_process = kwargs.pop("n_process", mp.cpu_count())

    if multiprocessing:

        m = mp.Manager()
        result_queue = m.Queue()

        n_combinations = len(tuples)

        combinations_to_try = mp.Queue()

        for i in range(n_combinations):
            combinations_to_try.put((i, tuples[i][0], tuples[i][1]))

        lock = mp.Lock()

        jobs = []
        for w in range(n_process):
            p = mp.Process(target=_transformer_worker,
                           args=(dataset, lock, combinations_to_try, result_queue),
                           kwargs=kwargs)
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        repr_arr = np.zeros(n_combinations, dtype=object)
        failed_arr = np.zeros(n_combinations, dtype=int)

        num_res = result_queue.qsize()
        while num_res > 0:
            i, repr_matrix, failed = result_queue.get()
            repr_arr[i] = repr_matrix
            failed_arr[i] = failed
            num_res -= 1
    else:
        n_combinations = len(tuples)
        repr_arr = np.zeros(n_combinations, dtype=object)
        failed_arr = np.zeros(n_combinations, dtype=int)
        transf = BOPTransformer(**kwargs)
        for i, pair in enumerate(tuples):
            win, wl = float(pair[0]), int(pair[1])
            transf["window"] = win
            transf["word_length"] = wl
            print("processing params [win %.3f, wl %d]" % (win, wl))
            ini = time.time()
            bop_repr, failed = transf.transform_dataset(dataset)
            end = time.time()
            print("processed params [win %.3f, wl %d] [time: %.3f seconds]" % (win, wl,
                                                                   round(end-ini, 3)))
            repr_arr[i] = bop_repr
            failed_arr[i] = failed

    return repr_arr, failed_arr


def _reducer_worker(repr_arr: np.ndarray, labels: np.ndarray,
                    lock: mp.synchronize.Lock,
                    combinations_to_try: mp.queues.Queue,
                    result_queue: mp.queues.Queue, **kwargs):
    try:
        scale = kwargs.pop("scale", True)
        n_splits = kwargs.pop("n_splits", 10)
        similarity_type = kwargs.pop("similarity_type", "cosine")
        scaler = StandardScaler()
        while True:
            try:
                lock.acquire()
                i, combis = combinations_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                pca = PCAReducer(**kwargs)
                combined_repr = BOPSparseRepresentation()
                combined_repr.copy_from(repr_arr[combis[0]])
                if len(combis) > 1:
                    for i in range(1, len(combis)):
                        combined_repr.hstack_repr(repr_arr[combis[i]])
                if scale:
                    combined_repr.set_scaler(scaler)
                combined_reduced_repr = pca.fit_transform(combined_repr)
                n_com = combined_reduced_repr.shape[1]
                acc = _pca_cross_validation(combined_repr, labels, n_com, n_splits,
                                            similarity_type=similarity_type, scale=scale)

                result_queue.put((i, n_com, acc, pca))
    except Exception as e:
        print("worker '%s' failed with error: " % mp.current_process().name, e)


def _pca_cross_validation(combined_repr: BOPSparseRepresentation,
                          labels: np.ndarray, n_com: int, n_splits: int,
                          scale=True, similarity_type="cosine"):

    cv_splitter = StratifiedKFold(n_splits=n_splits)
    vectors = combined_repr.to_array()
    scaler = StandardScaler()
    pca = PCA(n_components=n_com)

    real = []
    pred = []
    for train_index, test_index in cv_splitter.split(vectors, labels):
        X_train, X_test = vectors[train_index], vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if scale:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        if similarity_type == "cosine":
            y_pred = predict_by_cosine(X_train_reduced, y_train, X_test_reduced)
        elif similarity_type == "euclidean":
            y_pred = predict_by_euclidean(X_train_reduced, y_train, X_test_reduced)
        else:
            raise ValueError("similarity type '%s' unknown" % similarity_type)

        real.append(y_test)
        pred.append(y_pred)

    return balanced_accuracy_score(real, pred)


def reduce_cv_bop(repr_arr: np.ndarray, labels: np.ndarray, combinations: list,
                  multiprocessing=True, **kwargs):
    n_process = kwargs.pop("n_process", mp.cpu_count())

    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(combinations)

    combinations_to_try = mp.Queue()

    for i in range(n_combinations):
        combinations_to_try.put((i, combinations[i]))

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=_transformer_worker,
                       args=(repr_arr, labels, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    pca_arr = np.zeros(n_combinations, dtype=object)
    n_com_arr = np.zeros(n_combinations, dtype=int)
    acc_arr = np.zeros(n_combinations, dtype=float)

    num_res = result_queue.qsize()
    while num_res > 0:
        i, n_com, acc, pca = result_queue.get()
        pca_arr[i] = pca
        n_com_arr[i] = n_com
        acc_arr[i] = acc
        num_res -= 1

    return pca_arr, n_com_arr, acc_arr