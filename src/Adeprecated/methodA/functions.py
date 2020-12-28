import multiprocessing as mp
import queue
from .transformer2 import BOPTransformer
from .representation import BOPSparseRepresentation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from src.Adeprecated.predictors import predict_by_cosine, predict_by_euclidean
from sklearn.metrics import balanced_accuracy_score
import time


def _transformer_worker(dataset: np.ndarray, lock: mp.synchronize.Lock,
                        combinations_to_try: mp.queues.Queue,
                        result_queue: mp.queues.Queue, **kwargs):
    try:
        transf = BOPTransformer(**kwargs)
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
                transf["window"] = win
                transf["word_length"] = wl
                ini = time.time()
                bop_repr, failed = transf.transform_dataset(dataset)
                end = time.time()
                print("'%s' processed params [win %.3f, wl %d] [time: %.3f seconds]" % (mp.current_process().name,
                                                                                        win, wl,
                                                                                   round(end - ini, 3)))
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


def _reducer_worker(repr_arr: np.ndarray, labels: np.ndarray, n_components: int,
                    lock: mp.synchronize.Lock,
                    combinations_to_try: mp.queues.Queue,
                    result_queue: mp.queues.Queue, **kwargs):
    try:
        scale = kwargs.pop("scale", False)
        n_splits = kwargs.pop("n_splits", 10)
        similarity_type = kwargs.pop("similarity_type", "cosine")
        reducer_type = kwargs.pop("reducer_type", "lsa")  # not used for now
        scaler = StandardScaler()
        while True:
            try:
                lock.acquire()
                iii, combis, idxs = combinations_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                # print(combis)

                combined_repr = BOPSparseRepresentation()
                combined_repr.copy_from(repr_arr[idxs[0]])
                failed0 = combined_repr.count_failed()
                if len(idxs) > 1:
                    for i in range(1, len(idxs)):
                        combined_repr.hstack_repr(repr_arr[idxs[i]])
                        failed1 = combined_repr.count_failed()
                        assert failed1 >= failed0
                        failed0 = failed1
                # if scale:
                #     combined_repr.set_scaler(scaler)
                failed = combined_repr.count_failed()
                n = combined_repr.vector.shape[0]
                n0 = repr_arr[idxs[0]].vector.shape[0]
                if n != n0:
                    print("something wrog, n before {}, n after {}".format(n0, n))
                    raise ValueError()
                success_rate = (n - failed) / n
                if success_rate < 0.8:
                    print("success rate too low {} < {} for {} combis, iteration dropped".format(success_rate, 0.8, idxs))
                    continue

                combined_repr.sample_wise_norm()

                n_com = min(n_components, combined_repr.vector.shape[1]-1)
                reducer = TruncatedSVD(n_components=n_com, n_iter=10)
                combined_reduced_repr = reducer.fit_transform(combined_repr.vector)
                n_com = combined_reduced_repr.shape[1]
                # print("{} for combined params {} using {} components has {} variance ratio".format(reducer_type.upper(), combis, n_com, np.sum(reducer.explained_variance_ratio_)))

                acc = _reducer_cross_validation_no_scale(combined_repr, labels, n_com, n_splits, reducer_type,
                                            similarity_type=similarity_type)
                print("acc for [{:.3f} success rate, {} components, {} method, {:.3f} variance ratio, {} combis] -> {:.2f}".format(
                    success_rate,  n_com,  reducer_type.upper(), round(np.sum(reducer.explained_variance_ratio_), 3),
                    idxs, acc))

                result_queue.put((iii, acc, failed, n_com))
    except Exception as e:
        print("worker '%s' failed with error: " % mp.current_process().name, e)


def _reducer_cross_validation_no_scale(combined_repr: BOPSparseRepresentation,
                          labels: np.ndarray, n_com: int, n_splits: int, reducer_type: str,
                                       similarity_type="cosine"):
    cv_splitter = StratifiedKFold(n_splits=n_splits)
    vectors = combined_repr.vector
    # pca = PCA(n_components=n_com)
    reducer = TruncatedSVD(n_components=n_com, n_iter=20)

    real = []
    pred = []
    for train_index, test_index in cv_splitter.split(vectors, labels):
        X_train, X_test = vectors[train_index], vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)

        if similarity_type == "cosine":
            y_pred = predict_by_cosine(X_train_reduced, y_train, X_test_reduced)
        elif similarity_type == "euclidean":
            y_pred = predict_by_euclidean(X_train_reduced, y_train, X_test_reduced)
        else:
            raise ValueError("similarity type '%s' unknown" % similarity_type)

        real.extend(y_test)
        pred.extend(y_pred)

    return balanced_accuracy_score(real, pred)


def reduce_cv_bop(repr_arr: np.ndarray, labels: np.ndarray, combinations: list, combis_idx: list, n_components: int,
                  multiprocessing=True, **kwargs):
    n_process = kwargs.pop("n_process", mp.cpu_count())
    n_process = 1

    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(combinations)

    combinations_to_try = mp.Queue()

    for i in range(n_combinations):
        combinations_to_try.put((i, combinations[i], combis_idx[i]))

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=_reducer_worker,
                       args=(repr_arr, labels, n_components, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    n_com_arr = np.zeros(n_combinations, dtype=int)
    acc_arr = np.zeros(n_combinations, dtype=float)
    failed_arr = np.zeros(n_combinations, dtype=int)

    num_res = result_queue.qsize()
    while num_res > 0:
        i, acc, failed, n_com = result_queue.get()
        # print(acc, i)
        n_com_arr[i] = n_com
        acc_arr[i] = acc
        failed_arr[i] = failed
        num_res -= 1

    return n_com_arr, acc_arr, failed_arr
