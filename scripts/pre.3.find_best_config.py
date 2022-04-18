import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
import time
from ibopf.preprocesing import gen_dataset_from_h5,  rearrange_splits, get_ibopf_plasticc_path
from ibopf.cross_validation import cv_mmm_bopf,  load_bopf_from_quantity_search
from ibopf.pipelines.method import IBOPF
import argparse
from multiprocessing import cpu_count
import pandas as pd

'''
Script used to find best Multi-resolution Multi-quantity Multi-variate Bag-of-Patterns Features
using a grid search algorithm

The script is intended to work only with PLaSTiCC dataset

The parameters window width (win) and word length (wl) are fixed for this script

'''

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]  # bands on PLaSTiCC dataset


def get_fixed_arguments():
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

    pre_load_bopf = True
    return doc_kwargs, lsa_kwargs, pre_load_bopf


def get_dataset_variables(dataset_name, select_survey=None):
    # read dataset
    dataset, labels_, metadata, split_folds = gen_dataset_from_h5(dataset_name, bands=_BANDS, num_folds=5, select_survey=select_survey)
    split_folds = rearrange_splits(split_folds)
    classes = np.unique(labels_)
    print(len(labels_))
    N = int(np.mean([len(ts[0]) * 2 for ts in dataset]))

    return dataset, labels_, metadata, split_folds, classes, N


def check_or_create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_files_and_folders(c, select_survey=None):
    data_path = get_ibopf_plasticc_path()

    resolution_search_main_directory = os.path.join(data_path, "resolution_search")
    check_or_create_folder(resolution_search_main_directory)
    resolution_search_base_result = os.path.join(data_path, "res_search_base_res")
    check_or_create_folder(resolution_search_base_result)
    resolution_search_directory = os.path.join(resolution_search_main_directory, c.lower())
    check_or_create_folder(resolution_search_directory)
    name = "%s_resolution_search" % c.lower()
    if select_survey is not None:
        name += "_%s" % select_survey
    method_sub_directory = os.path.join(resolution_search_directory, name)
    check_or_create_folder(method_sub_directory)

    name = "optimal_config_%s" % c.lower()
    if select_survey is not None:
        name += "_%s" % select_survey
    name += ".json"
    config_file = os.path.join(data_path, name)

    return resolution_search_directory, method_sub_directory, config_file, resolution_search_base_result


def run_script(dataset_name, q_code, q_search_path, q_search_cv_results,
               top_k, resolution_max, alpha, C, timestamp, n_jobs, select_survey=None):

    dataset, labels_, metadata, split_folds, classes, N = get_dataset_variables(dataset_name, select_survey=select_survey)
    doc_kwargs, lsa_kwargs, pre_load_bopf = get_fixed_arguments()
    drop_zero_variance = C.lower() == "lsa"  # drop zero variance doesnt work for MANOVA
    resolution_search_directory, method_sub_directory, config_file,  out_base_bopf_path = get_files_and_folders(C, select_survey=select_survey)


    # get pipeline
    method = IBOPF(alpha=alpha, Q_code=q_code, C=C, lsa_kw=lsa_kwargs,
                   doc_kw=doc_kwargs, N=N, n_jobs=n_jobs,
                   drop_zero_variance=drop_zero_variance)


    # pre-load saved base bopf
    print("LOADING PRECOMPUTED BASE BOPF...")
    if args.cv_smm_again is None:
        cv_smm_bopf_results = load_bopf_from_quantity_search(q_search_path,
                                                         q_search_cv_results,
                                                         method.quantities_code())
        wins = None
    else:
        cv_smm_bopf_results = None
        wins = np.append(np.array([4, 6, 8, 10, 14, 18, 25]), np.logspace(np.log10(30), np.log10(1000), 20))

    wls = [1, 2, 3]

    R, _, optimal_acc = cv_mmm_bopf(
        dataset, labels_, method, cv=split_folds, resolution_max=resolution_max,
        top_k=top_k, out_path=method_sub_directory, n_jobs=n_jobs,
        cv_smm_bopf_results=cv_smm_bopf_results, drop_zero_variance=drop_zero_variance,
        timestamp=timestamp, out_base_bopf_path=out_base_bopf_path, wls=wls, wins=wins)

    return R, optimal_acc, method, config_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to find best combination of quantities on.'
    )
    parser.add_argument(
        "multi_quantity_resume_file",
        help="The resume with the search of the optimal multi-quantity combination"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="the top K single-resolution representations to try on this multi-resolution search"
    )
    parser.add_argument(
        "--resolution_max",
        type=int,
        default=4,
        help="The maximum number of resolutions to include in the optimal multi-resolution combination"
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=4,
        help="alphabet size to use during the search"
    )

    parser.add_argument(
        "--compact_method",
        type=str,
        default="LSA",
        help="The compact method to use, options are: LSA or MANOVA"
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
    parser.add_argument("--cv_smm_again", type=str, default=None)

    parser.add_argument("--split_surveys", type=str, default=None)
    args = parser.parse_args()

    if args.n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = args.n_jobs

    ini = time.time()
    # we try the top 2 quantities combinations and save the configuration for the best
    quantity_search_resume = pd.read_csv(args.multi_quantity_resume_file, index_col=None)
    quantity_search_resume = quantity_search_resume.sort_values("cv_mean", ascending=False)

    top1 = quantity_search_resume[quantity_search_resume["quantity"] == "(TrMm-MmMn-MmMx-TrMn)"].iloc[0]
    if args.split_surveys:
        R1, optimal_acc1, method1, config_file1 = run_script(
            args.dataset, top1.quantity, top1.q_search_path, top1.cv_results_file, args.top_k,
            args.resolution_max, args.alpha, args.compact_method, args.timestamp, n_jobs, select_survey="ddf")

        R2, optimal_acc2, method2, config_file2 = run_script(
            args.dataset, top1.quantity, top1.q_search_path, top1.cv_results_file, args.top_k,
            args.resolution_max, args.alpha, args.compact_method, args.timestamp, n_jobs, select_survey="wdf")

    else:
        R1, optimal_acc1, method1, config_file1 = run_script(
            args.dataset, top1.quantity, top1.q_search_path, top1.cv_results_file, args.top_k,
            args.resolution_max, args.alpha, args.compact_method, args.timestamp, n_jobs)

    # top2 = quantity_search_resume.iloc[1]
    # R2, optimal_acc2, method2, config_file2 = run_script(
    #     args.dataset, top2.quantity, top2.q_search_path, top2.cv_results_file, args.top_k,
    #     args.resolution_max, args.alpha, args.compact_method, args.timestamp, n_jobs)

    # if optimal_acc2 > optimal_acc1:
    #     R1 = R2
    #     optimal_acc1 = optimal_acc2
    #     method1 = method2
    #     config_file1 = config_file2

    end = time.time()

    print("ELAPSE TIME: %.3f secs (%.4f Hours)" % ((end - ini), (end - ini) / 3600))

    try:
        log_file_finder = os.path.join(
            "..", "data", "configs_results_new", "%s_multi_ress_search.txt" % args.compact_method.lower())
        f = open(log_file_finder, "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("compact method: %s\n" % args.compact_method)
        f.write("alphabet size: %s\n" % str(args.alpha))
        f.write("quantities: %s\n" % args.q_code)
        f.write("resolutions: %s\n" % repr(R1))
        f.write("optimal acc_cv: %s\n" % str(optimal_acc1))
        f.write("timestamp: %s\n" % str(args.timestamp))
        f.write("split_surveys: %s\n" % "True" if args.split_surveys is not None else "False")
        f.write("elapse time: %.3f secs (%.4f Hours)\n" % ((end - ini), (end - ini) / 3600))
        f.write("comment: run with only forward slider and fixed index\n")  # this change
        f.write("++++++++++++++++++++++++++++++++\n")
        f.close()
    except Exception as e:
        print("failed to write log file, error: ", e)

    # R = [(115.691, 1), (393.505, 2), (642.112, 1), (642.112, 3)]
    # R for new implementation, LSA and alpha=4
    # R = [(win:wl)-(122.649:1)-(367.244:2)-(429.533:1)-(122.649:2)]  (acc: 0.480)
    # R for new implementation, LSA, alpha=4, and forward/backward segmentator
    # R = [(win:wl)-(89.656:1)-(587.597:2)-(687.260:2)-(429.533:1)] (acc: 0.459)
    # R for new imp, LSA, alpha=2, and only forward slicer/segmentator
    # R = ?

    # best config comb triple-q
    # FINAL BEST CONFIG:  [(406.48199999999997, 1), (63.172, 1)] , acc:  0.5017615943950111
    # best config comb double-q
    # [(110.428, 1), (589.8530000000001, 1)] , acc:  0.5040139362331313

    method1.config_to_json(config_file1)
    # TODO: CORRER DENUEVO PERO PARA ALPHA=6 Y WL= {1, 2}.
    # call script with python -m sklearnex 2.find_best_config.py

