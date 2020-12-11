import multiprocessing as mp
from .transformer import BOPFTransformer
from .classify import simple_train_test_classify
from .class_vectors import compute_class_centroids, compute_class_tf_idf
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score
import queue
import numpy as np
from collections import defaultdict


def bopf_transformer_worker(train_set, y1, y2, prev_centroid, prev_tf_idf, lock, comb_to_try, r_queue, **kwargs):
    try:
        transf = BOPFTransformer(**kwargs)
        transf.set_full_alphabet(transf['alph_size'])
        transf.logger.info("start BOPFTransformer on worker '%s'" % mp.current_process().name)
        while True:
            try:
                lock.acquire()
                win, wl, i = comb_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                transf.alphabet = transf["full_alphabet"][:transf.get_alph_size()]
                # if transf["verbose"]:
                #     transf.logger.info("computing BagOfPatter for [{}, {}] on worker '{}'".format(win, wl,
                #                                                                                   mp.current_process().name))
                transf.bop(train_set[0], train_set[1], wl, win)
                _ = transf.count_failed()
                fvalues, classes, c, class_count, positions = transf.anova(len(train_set[0]), train_set[2], wl)
                _, limit = transf.reduce_zeros(fvalues)
                if limit == 0:
                    transf.logger.info("[DROPPING] dropping pair [win={}, wl={}] since ANOVA  gives all 0".format(win,
                                                                                                                  wl))
                else:
                # transf.logger.info("[CV_REDUCE] looking for best k features, limit: {} [worker '{}']".format(limit,
                #                                                                                  mp.current_process().name))


                    # fea_num_c, bacc_c, fea_num_tfidf, bacc_tfidf = transf.cv_reduce_best_fea_num(limit, np.array(train_set[2]),
                    #                                                                          prev_vector,
                    #                                                                          n_splits=n_splits)

                    res_centroid, res_tf_idf, y1out, y2out = transf.cv_reduce_best_fea_num2(limit, train_set[2],
                                                                                            positions, class_count,
                                                                                            y1in=y1, y2in=y2)
                    if prev_centroid[0] == win and prev_centroid[1] == wl:
                        res_centroid = [-np.inf, -np.inf]
                    if prev_tf_idf[0] == win and prev_tf_idf[1] == wl:
                        res_tf_idf = [-np.inf, -np.inf]
                    transf.logger.info("[CENTROID] bacc: {} with k={} || [TFIDF] bacc: {} with k={} (win={}, wl={})".format(
                        res_centroid[1], res_centroid[0], res_tf_idf[1], res_tf_idf[0], win, wl
                    ))
                    r_queue.put((i, wl, win, res_centroid[0], res_tf_idf[0], res_centroid[1], res_tf_idf[1],
                                 y1out, y2out))
    except Exception as e:
        print("worker failed with error:", e)
        transf = None
    finally:
        # if transf is not None:
        #     transf.logger.info("worker '%' DONE" % mp.current_process().name)
        # else:
        print("worker '%s' DONE" % mp.current_process().name)


def bopf_transformer_mp(train_set,  tuples, n_splits=5, n_process="default", y1=None, y2=None,
                        prev_centroid=None, prev_tf_idf=None, **kwargs):
    if n_process == "default":
        n_process = mp.cpu_count()

    if prev_tf_idf is None:
        prev_tf_idf = [-1, -1]
    if prev_centroid is None:
        prev_centroid = [-1, -1]
    # if prev_vector is None:
    #     prev_vector = {"centroid": None, "tf_idf": None}

    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(tuples)
    combinations_to_try = mp.Queue()

    for i in range(n_combinations):
        combinations_to_try.put((tuples[i][0], tuples[i][1], i))

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=bopf_transformer_worker,
                       args=(train_set, y1, y2, prev_centroid, prev_tf_idf, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    output_dict = defaultdict(list)
    num_res = result_queue.qsize()
    while num_res > 0:
        i, wl, win, fea_num_centroid, fea_num_tf_idf, bacc_centroid, bacc_tf_idf, y1, y2 = result_queue.get()
        output_dict["i"].append(i)
        output_dict["wl"].append(wl)
        output_dict["win"].append(win)
        output_dict["fea_num_centroid"].append(fea_num_centroid)
        output_dict["fea_num_tf_idf"].append(fea_num_tf_idf)
        output_dict["bacc_centroid"].append(bacc_centroid)
        output_dict["bacc_tf_idf"].append(bacc_tf_idf)
        output_dict["y1"].append(y1)
        output_dict["y2"].append(y2)
        num_res -= 1

    return output_dict


def bopf_transformer_separated_worker(train_set, y_in, prev_method, method, only_vectors, test_fvalues, lock, comb_to_try, r_queue, **kwargs):
    try:
        transf = BOPFTransformer(**kwargs)
        transf.set_full_alphabet(transf['alph_size'])
        transf.logger.info("start BOPFTransformer on worker '%s'" % mp.current_process().name)
        while True:
            try:
                lock.acquire()
                win, wl, i = comb_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                transf.alphabet = transf["full_alphabet"][:transf.get_alph_size()]
                transf.bop(train_set[0], train_set[1], wl, win)
                # print("bop done")
                failed = transf.count_failed()
                if failed > len(train_set[0]) // 20:
                    continue
                if test_fvalues is not None:
                    # transf.logger.info("evaluation process")
                    # transf.logger.info("[{}] bop size before zero reduction: {}, i: {}".format(mp.current_process().name,
                    #                                                                            transf.count_vector.shape,
                    #                                                                            i))
                    _, limit = transf.reduce_zeros(test_fvalues[i], verbose=True)
                    # transf.logger.info("[{}] bop size after zero reduction: {}".format(mp.current_process().name, transf.count_vector.shape))
                    if only_vectors:
                        classes, c, class_count, positions = transf.get_class_count(len(train_set[2]), train_set[2])
                        if method == "centroid":
                            vec = transf.get_centroid_vectors(limit, train_set[2], positions, class_count)
                        else:
                            vec = transf.get_tf_idf_vectors(limit, train_set[2], positions)
                    else:
                        if method == "centroid":
                            vec = transf.count_vector
                        else:
                            vec = transf.prepare_count_vectors_for_tf_idf()
                    r_queue.put((i, wl, win, None, None, None, vec, None, failed))
                    continue
                fvalues, classes, c, class_count, positions = transf.anova(len(train_set[0]), train_set[2], wl)
                _, limit = transf.reduce_zeros(fvalues)
                if limit == 0:
                    transf.logger.info("[DROPPING] dropping pair [win={}, wl={}] since ANOVA  gives all 0".format(win,
                                                                                                                  wl))
                else:
                    if method == "centroid":
                        if not only_vectors:
                            res, yout, vec = transf.cv_fea_num_centroid(limit, train_set[2], positions, class_count,
                                                               y1in=y_in)
                        else:
                            res, yout = (None, None), None
                            vec = transf.get_centroid_vectors(limit, train_set[2], positions, class_count)

                    else:
                        if not only_vectors:
                            res, yout = transf.cv_fea_num_tf_idf(limit, train_set[2], positions,
                                                               y2in=y_in)
                            vec = None
                        else:
                            res, yout= (None, None), None
                            vec = transf.get_tf_idf_vectors(limit, train_set[2], positions)

                    if prev_method[0] == win and prev_method[1] == wl:
                        res = [-np.inf, -np.inf]

                    if not only_vectors:
                        transf.logger.info("[method:{}, win={}, wl={}] bacc: {} with k={}, len_fvalues={}".format(
                            method.upper(), win, wl, res[1], res[0], len(fvalues)
                        ))
                    r_queue.put((i, wl, win, res[0], res[1], yout, vec, fvalues, failed))

    except Exception as e:
        print("worker failed with error:", e)
        transf = None
    finally:
        # if transf is not None:
        #     transf.logger.info("worker '%' DONE" % mp.current_process().name)
        # else:
        print("worker '%s' DONE" % mp.current_process().name)


def bopf_transformer_separated_mp(train_set,  tuples, method="centroid", n_process="default", yin=None,
                        prev_method=None, only_vectors=False, test_fvalues=None, **kwargs):
    if n_process == "default":
        n_process = mp.cpu_count()

    if prev_method is None:
        prev_method = [-1, -1]

    m = mp.Manager()
    result_queue = m.Queue()

    n_combinations = len(tuples)
    combinations_to_try = mp.Queue()

    for i in range(n_combinations):
        combinations_to_try.put((tuples[i][0], tuples[i][1], i))

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=bopf_transformer_separated_worker,
                       args=(train_set, yin, prev_method, method, only_vectors, test_fvalues, lock, combinations_to_try, result_queue),
                       kwargs=kwargs)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    output_dict = defaultdict(list)
    num_res = result_queue.qsize()
    while num_res > 0:
        i, wl, win, fea_num, bacc, yout, vec, fvalues, failed = result_queue.get()
        output_dict["i"].append(i)
        output_dict["wl"].append(wl)
        output_dict["win"].append(win)
        output_dict["fea_num"].append(fea_num)
        output_dict["bacc"].append(bacc)
        output_dict["yout"].append(yout)
        output_dict["vec"].append(vec)
        output_dict["fvalues"].append(fvalues)
        output_dict["failed"].append(failed)
        num_res -= 1

    return output_dict