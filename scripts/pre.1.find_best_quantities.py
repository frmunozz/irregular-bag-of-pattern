# -*- coding: utf-8 -*-
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
import time
from scipy import sparse
from ibopf.preprocesing import gen_dataset_from_h5, rearrange_splits, get_mmbopf_plasticc_path
from ibopf.cross_validation import cv_smm_bopf
from ibopf.pipelines.method import IBOPF
import pickle
import argparse
from multiprocessing import cpu_count


'''
Script used to find best Single-resolution Multi-quantity Multi-variate Bag-of-Patterns Features
using a grid search algorithm

The script is intended to work only with PLaSTiCC dataset

'''

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]  # bands on PLaSTiCC dataset


class ConfigData(object):

    def __init__(self, **kwargs):
        """
        simple class to store all the used parameters
        :param kwargs: the parameters to store
        """
        self._kwargs = kwargs

    def add(self, **kwargs):
        self._kwargs.update(kwargs)

    def __getattribute__(self, item):
        if item == "_kwargs" or item == "add":
            return super(ConfigData, self).__getattribute__(item)
        if item in self._kwargs.keys():
            return self._kwargs[item]
        else:
            raise ValueError("If called, attribute '%s' should be defined" % item)


def independent_comb_search(Q_arr_sorted, Q_acc_sorted, max_comb, comb_q_file, data_folder, D: ConfigData, method):

    comb_Q = [Q_arr_sorted[0]]
    i = 1
    best_comb_acc = Q_acc_sorted[0]
    while i < max_comb:
        best_iter_acc = best_comb_acc
        best_iter_q = None
        for q in Q_arr_sorted:
            if q not in comb_Q:
                # try this comb
                Q = comb_Q.copy()
                Q.append(q)
                method.Q = Q

                data_mr_repr, cv_results, result_lists, best_acc, best_data = cv_smm_bopf(
                    D.dataset, D.labels_, D.wins, D.wls, method, cv=D.split_folds, n_jobs=D.n_jobs,
                    drop_zero_variance=D.drop_zero_variance, outfile=comb_q_file, C=D.C,
                    data_folder=data_folder
                )
                if best_acc > best_iter_acc:
                    best_iter_acc = best_acc
                    best_iter_q = q
        if best_iter_acc > best_comb_acc:
            diff = best_iter_acc - best_comb_acc
            comb_Q.append(best_iter_q)
            i += 1
            best_comb_acc = best_iter_acc
            if diff < 0.01:
                # the improvement is not higher than 1%, so we stop
                break
        else:
            break
    return comb_Q, best_comb_acc


def check_if_new(arr, v):
    for x in arr:
        if all([y in v for y in x]):
            return False
    return True


def dependent_comb_search(selected_q_arr, single_q_arr, quantity_file, data_folder, D: ConfigData, method):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    new_q_arr = []
    new_q_acc = []
    for first_q in selected_q_arr:
        if isinstance(first_q, str):
            first_q = [first_q]
        for second_q in single_q_arr:
            if second_q not in first_q:
                q_fusion = first_q + [second_q]
                if check_if_new(new_q_arr, q_fusion):
                    new_q_arr.append(q_fusion)
                    Q = [q_fusion]
                    method.Q = Q
                    data_mr_repr, cv_results, result_lists, best_acc, best_data = cv_smm_bopf(
                        D.dataset, D.labels_, D.wins, D.wls, method, cv=D.split_folds, n_jobs=D.n_jobs,
                        drop_zero_variance=D.drop_zero_variance, outfile=quantity_file, C=D.C
                    )
                    for k1, v1 in data_mr_repr.items():
                        for k2, v2 in v1.items():
                            data_repr_i = data_mr_repr[k1][k2]
                            sparse.save_npz(os.path.join(
                                data_folder, method.quantities_code()[1:-1] + "_%d_%.3f.npz" % (k2, k1)), data_repr_i)
                    new_q_acc.append(best_acc)

    idxs_top = np.argsort(new_q_acc)[::-1]
    sorted_q_arr = []
    sorted_acc_arr = []
    for i in idxs_top:
        sorted_q_arr.append(new_q_arr[i])
        sorted_acc_arr.append(new_q_acc[i])

    return sorted_q_arr, sorted_acc_arr


def single_q_search(single_Q_arr, D: ConfigData, pipeline):
    single_quantity_file = os.path.join(D.out_path, "single_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "single_quantity_%s_data" % D.C.lower())
    _header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(single_quantity_file, "a+")
    f.write(_header + "\n")
    f.close()

    sorted_q_arr, sorted_acc_arr = dependent_comb_search([[]], single_Q_arr, single_quantity_file, data_folder, D,
                                                         pipeline)

    out_dict = {"sorted_q_arr": sorted_q_arr, "sorted_acc_arr": sorted_acc_arr}
    single_q_resume_file = os.path.join(D.out_path, "single_q_resume_%s_%s.pkl" % (D.C.lower(), D.timestamp))
    pickle.dump(out_dict, open(single_q_resume_file, "wb"))

    print(":::: Top single-Q is: %s (acc: %.3f)" % (repr(sorted_q_arr[0]), sorted_acc_arr[0]))

    return sorted_q_arr, sorted_acc_arr


def double_q_search(selected_single_q_arr, single_Q_arr, D: ConfigData, method):
    double_quantity_file = os.path.join(D.out_path, "double_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "double_quantity_%s_data" % D.C.lower())

    header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(double_quantity_file, "a+")
    f.write(header + "\n")
    f.close()
    sorted_q_arr, sorted_acc_arr = dependent_comb_search(selected_single_q_arr, single_Q_arr,
                                                         double_quantity_file, data_folder, D, method)
    out_dict = {"sorted_q_arr": sorted_q_arr, "sorted_acc_arr": sorted_acc_arr}
    double_q_resume_file = os.path.join(D.out_path, "double_q_resume_%s_%s.pkl" % (D.C.lower(), D.timestamp))
    pickle.dump(out_dict, open(double_q_resume_file, "wb"))

    print(":::: Top pair-Q is: %s (acc: %.3f)" % (repr(sorted_q_arr[0]), sorted_acc_arr[0]))

    return sorted_q_arr, sorted_acc_arr


def triple_q_search(selected_double_q_arr, single_q_arr, D: ConfigData, method):
    triple_quantity_file = os.path.join(D.out_path, "triple_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "triple_quantity_%s_data" % D.C.lower())
    _header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(triple_quantity_file, "a+")
    f.write(_header + "\n")
    f.close()

    sorted_q_arr, sorted_acc_arr = dependent_comb_search(selected_double_q_arr, single_q_arr,
                                                         triple_quantity_file, data_folder, D, method)

    out_dict = {"sorted_q_arr": sorted_q_arr, "sorted_acc_arr": sorted_acc_arr}
    triple_q_resume_file = os.path.join(D.out_path, "triple_q_resume_%s_%s.pkl" % (D.C.lower(), D.timestamp))
    pickle.dump(out_dict, open(triple_q_resume_file, "wb"))

    print(":::: Top trio-Q is: %s (acc: %.3f)" % (repr(sorted_q_arr[0]), sorted_acc_arr[0]))

    return sorted_q_arr, sorted_acc_arr


def single_q_comb_search(single_q_arr_sorted, single_q_acc_sorted, max_comb, D: ConfigData, method):
    comb_single_quantity_file = os.path.join(D.out_path, "comb_single_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "single_quantity_%s_data" % D.C.lower())
    _header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(comb_single_quantity_file, "a+")
    f.write(_header + "\n")
    f.close()

    comb_q, best_comb_acc = independent_comb_search(single_q_arr_sorted, single_q_acc_sorted,
                                                    max_comb, comb_single_quantity_file, data_folder, D, method)

    print(":::: Top combination of single-Q is: %s (acc: %.3f)" % (repr(comb_q), best_comb_acc))

    return comb_q, best_comb_acc


def double_q_comb_search(pair_q_arr_sorted, pair_q_acc_sorted, max_comb, D: ConfigData, method):
    comb_double_quantity_file = os.path.join(D.out_path, "comb_double_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "double_quantity_%s_data" % D.C.lower())
    _header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(comb_double_quantity_file, "a+")
    f.write(_header + "\n")
    f.close()

    comb_q, best_comb_acc = independent_comb_search(pair_q_arr_sorted, pair_q_acc_sorted,
                                                    max_comb, comb_double_quantity_file, data_folder, D, method)

    print(":::: Top combination of pair-Q is: %s (acc: %.3f)" % (repr(comb_q), best_comb_acc))

    return comb_q, best_comb_acc


def triple_q_comb_search(trio_q_arr_sorted, trio_q_acc_sorted, max_comb, D: ConfigData, pipeline):
    comb_triple_quantity_file = os.path.join(D.out_path, "comb_triple_quantity_%s_%s" % (D.C.lower(), D.timestamp))
    data_folder = os.path.join(D.out_path, "triple_quantity_%s_data" % D.C.lower())
    _header = "compact_method,quantity,win,wl,alpha,dropped,bopf_shape,valid_cv,cv_mean,cv_std,bopf_time,cv_time"
    f = open(comb_triple_quantity_file, "a+")
    f.write(_header + "\n")
    f.close()

    comb_q, best_comb_acc = independent_comb_search(trio_q_arr_sorted, trio_q_acc_sorted,
                                                    max_comb, comb_triple_quantity_file, data_folder, D, pipeline)

    print(":::: Top combination of trio-Q is: %s (acc: %.3f)" % (repr(comb_q), best_comb_acc))

    return comb_q, best_comb_acc


if __name__ == '__main__':
    main_ini = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to find best combination of quantities on.'
    )
    parser.add_argument(
        "--compact_method",
        type=str,
        default="LSA",
        help="The compact method to use, options are: LSA or MANOVA"
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=4,
        help="alphabet size to use during the search"
    )

    parser.add_argument(
        "--max_power",
        type=int,
        default=8,
        help="max power for the vocabulary expressed as V=alpha^(power)"
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
    )
    args = parser.parse_args()

    if args.alpha ** args.max_power <= 4 ** 6:
        raise ValueError("need to specify a power higher than 4^6")

    config = ConfigData()

    data_path = get_mmbopf_plasticc_path()
    out_main_path = os.path.join(data_path, "quantity_search")
    if not os.path.exists(out_main_path):
        os.mkdir(out_main_path)
    out_path = os.path.join(out_main_path, args.compact_method.lower())
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    config.add(out_path=out_path, timestamp=args.timestamp)

    dataset, labels_, metadata, split_folds = gen_dataset_from_h5(args.dataset, bands=_BANDS, num_folds=5)
    split_folds = rearrange_splits(split_folds)
    classes = np.unique(labels_)
    print(len(labels_))
    # estimate spatial complexity (N)
    N = int(np.mean([len(ts[0]) * 2 for ts in dataset]))
    print("the estimated size of each time series is {} [4 bytes units] in average".format(N))
    config.add(dataset=dataset, labels_=labels_, split_folds=split_folds, classes=classes, N=N)

    # estimate max window of observation
    time_durations = np.array(
        [ts[1][-1] - ts[1][0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    max_window = mean_time + std_time

    doc_kwargs = {
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }

    lsa_kwargs = {  # scheme: ltc
        "class_based": False,  # options: True, False
        "normalize": "l2",  # options: None, l2
        "use_idf": True,  # options: True, False
        "sublinear_tf": True  # options: True, False
    }

    # C = "MANOVA"  # compact method
    C = args.compact_method
    if args.n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = args.n_jobs
    alpha = args.alpha
    drop_zero_variance = C.lower() == "lsa"  # drop zero variance only work with LSA
    config.add(doc_kwargs=doc_kwargs, lsa_kwargs=lsa_kwargs, C=C, alpha=alpha,
               n_jobs=n_jobs, drop_zero_variance=drop_zero_variance)

    wins = np.logspace(np.log10(30), np.log10(mean_time + std_time * 2), 20)
    config.add(wins=wins)
    method = IBOPF(alpha=alpha, C=C, lsa_kw=lsa_kwargs,
                   doc_kw=doc_kwargs, N=N, n_jobs=n_jobs,
                   drop_zero_variance=drop_zero_variance)

    """ TESTING SINGLE QUANTITY CASE """
    wls = [2, 3, 4, 5, 6]
    single_q = ["mean", "trend", "var", "min_max", "min", "max"]  # we discard the Count quantity
    config.add(wls=wls)

    single_q_arr_sorted, single_q_acc_sorted = single_q_search(single_q, config, method)

    """ TESTING COMBINATION OF SINGLE QUANTITIES CASE """
    # using as upper limit alpha^power features and each single-quantity can have up to
    # alpha^6 features, we can generate a combination of at most
    # alpha^(power-6) (with power > 6) single-quantities
    max_comb = min(len(single_q), args.alpha ** (args.max_power - 6))
    single_q_comb_search(single_q_arr_sorted, single_q_acc_sorted, max_comb, config, method)

    """ TESTING DOUBLE QUANTITY CASE """
    wls = [1, 2, 3]
    single_q = ["mean", "trend", "var", "min_max", "min", "max"]  # we discard the Count quantity
    selected_q = single_q_arr_sorted[:3]  # using top 5 single-quantity for double-quantity
    # at least 1 selected_q must be present on each computed pair of quantities
    config.add(wls=wls)

    double_q_arr_sorted, double_q_acc_sorted = double_q_search(selected_q, single_q, config, method)

    """double_q_resume_file = os.path.join(config.out_path, "double_q_resume_lsa_20210903-044949.pkl")
    out_dict = pickle.load(open(double_q_resume_file, "rb"))
    print(type(out_dict))
    print(out_dict.keys())
    double_q_arr_sorted = out_dict["sorted_q_arr"]
    double_q_acc_sorted = out_dict["sorted_acc_arr"]

    q_arr = []
    acc_arr = []
    for i in range(len(double_q_arr_sorted)):
        if check_if_new(q_arr, double_q_arr_sorted[i]):
            q_arr.append(double_q_arr_sorted[i])
            acc_arr.append(double_q_acc_sorted[i])

    double_q_arr_sorted = q_arr
    double_q_acc_sorted = acc_arr

    print(double_q_arr_sorted)
    print(double_q_acc_sorted)"""

    """ TESTING COMBINATION OF DOUBLE QUANTITIES CASE """
    # the max combination is given by an upper limit of alpha^power, where each pair of quantities
    # gives a vocabulary of at most alpha^(len(wls) * len(Q)),
    # allowing to produce a combination of alpha^(power-len(wls)*len(Q)) pairs, where len(Q)=2
    wls = [1, 2, 3]
    config.add(wls=wls)
    max_comb = min(6, args.alpha ** (args.max_power - (len(wls) * 2)))
    double_q_comb_search(double_q_arr_sorted, double_q_acc_sorted, max_comb, config, method)

    """ TESTING TRIPLE QUANTITY CASE """
    wls = [1, 2]
    single_q = ["mean", "trend", "var", "min_max", "min", "max"]  # we discard the Count quantity
    selected_q = double_q_arr_sorted[:3]  # using top-5 double-quantity for triple-quantity
    # at least 1 selected_q must be present on each computed triple-quantity
    config.add(wls=wls)

    triple_q_arr_sorted, triple_q_acc_sorted = triple_q_search(selected_q, single_q, config, method)

    """ TESTING COMBINATION OF TRIPLE QUANTITIES CASE """
    # the max combination is given by an upper limit of alpha^power, where each pair of quantities
    # gives a vocabulary of at most alpha^(len(wls) * len(Q)),
    # allowing to produce a combination of alpha^(power-len(wls)*len(Q)) pairs, where len(Q)=3
    wls = [1, 2]
    config.add(wls=wls)
    max_comb = min(6, args.alpha ** (args.max_power - (len(wls) * 3)))
    triple_q_comb_search(triple_q_arr_sorted, triple_q_acc_sorted, max_comb, config, method)
    main_end = time.time()

    try:
        f = open(os.path.join(out_path, "log.txt"), "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("script_name: find_best_quantities.py\n")
        f.write("compact_method: %s\n" % C),
        f.write("alpha: %d\n" % args.alpha),
        f.write("power: %s\n" % args.max_power),
        f.write("execution time: %.3f\n" % (main_end - main_ini))
        f.write("++++++++++++++++++++++++++++++++\n")
        f.close()
    except Exception as e:
        print("failed to write log file, error: ", e)

    # call script with python -m sklearnex find_best_quantities.py
