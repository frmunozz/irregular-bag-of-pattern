import os
import pdb
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from collections import defaultdict
import time
from scipy import sparse

from src.preprocesing import gen_dataset_from_h5, label_folds, rearrange_splits
from src.feature_extraction.text import MPTextGeneratorMultivariateCountWords
from src.feature_selection.select_k_best import GeneralSelectKTop
from src.feature_selection.analysis_of_variance import manova_rank
from src.neighbors.knn import flattened_ddtw, Flatten3Dto2D
from sklearn.neighbors import KNeighborsClassifier
from src.neighbors import KNeighborsClassifier as knnclassifier
from src.feature_extraction.vector_space_model import VSM
from sklearn.preprocessing import Normalizer, StandardScaler
from src.decomposition import LSA, PCA
from src.feature_extraction.centroid import CentroidClass

'''
This file presents a simple script to run a multi-resolution multi-quantity multi-variate bag-of-patterns features 
algorithm. The selected parameters are:

alphabet_size=6
quantities = [mean, trend, min_max]
IR_scheme= ltc (log TF-IDF normalized)

Optional parameters to be defined by user:

compact method = (LSA/MANOVA)
length of compact vector = (default)500

The script runs a search for the best multi-resolution configuration within the ranges:

word length = [1, 2, 3]
window width = logspace(log(10), log(mean_bandwidth + std_bandwidth), 20)
'''


# internal fixed parameters:
symbols = {
    "mean": "M",
    "std": "S",
    "trend": "T",
    "min_max": "m",
}
#######################################################################


# script fixed parameters:
_N_JOBS = 8
######################################################################


# method fixed parameters
doc_kwargs = {
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }

lsa_kwargs = {  # scheme: ltc
        "class_based": True, # options: True, False
        "normalize": "l2", # options: None, l2
        "use_idf": True, # options: True, False
        "sublinear_tf": True # options: True, False
    }

alphabet_size = 6
quantities = [["mean", "trend", "min_max"]]
############################################################################


# data fixed parameters
_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
############################################################################


def quantities_code(quantities):
    f_name = ""
    for q_i in quantities:
        for q_ii in q_i:
            f_name += symbols[q_ii]
        f_name += "-"
    return f_name


def get_all_single_representations(data, labels, wins, wls, alphabet_size, quantities, classes, lsa_kwargs, doc_kwargs,
                                   k=100, max_dropped="default", cv=5):
    q_code = quantities_code(quantities)
    x_repr = defaultdict(lambda: defaultdict(object))
    x_score = defaultdict(lambda: defaultdict(object))
    x_acc_list = []
    win_list = []
    wl_list = []
    for wl in wls:
        for win in wins:
            message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
            try:
                x_i = multi_quantity_representation(data, win, wl, alphabet_size, quantities, doc_kwargs)
                x_repr[win][wl] = x_i
                score, pipeline, _, dropped, shapes = general_cv_score(x_i, labels, classes, lsa_kwargs,
                                                                       k=k, max_dropped=max_dropped,
                                                                       message=message, cv=cv)
                if not (score is None and pipeline is None):
                    x_repr[win][wl] = x_i
                    x_score[win][wl] = (score, pipeline, dropped, shapes)
                    x_acc_list.append(np.mean(score))
                    win_list.append(win)
                    wl_list.append(wl)
            except Exception as e:
                print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))

    return x_repr, x_score, x_acc_list, win_list, wl_list


def multi_quantity_representation(data,  win, wl, alphabet_size, q, doc_kwargs):
    if len(q) > 1:
        x_arr = []
        for q_i in q:
            doc_kwargs["quantity"] = np.array(q_i)
            doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q_i))
            text_gen = MPTextGeneratorMultivariateCountWords(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                       opt_desc=", " + "-".join(q_i), **doc_kwargs)
            x_i = text_gen.fit_transform(data)
            x_arr.append(x_i)
        x = sparse.hstack(x_arr, format="csr")
    else:
        doc_kwargs["quantity"] = np.array(q[0])
        doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q[0]))
        text_gen = MPTextGeneratorMultivariateCountWords(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                   opt_desc=", " + "-".join(q[0]), **doc_kwargs)
        x = text_gen.fit_transform(data)

    # print("matrix shape: ", x.shape)
    return x


def general_cv_score(x, labels, classes, lsa_kwargs, k=100, max_dropped="default", message="X", cv=5):
    # print("cross_validation")
    ini = time.time()
    if max_dropped == "default":
        max_dropped = int(0.05 * len(labels))
    dropped = len(np.where(np.sum(x, axis=1) == 0)[0])
    if dropped > max_dropped:
        print("[{}] dropped because {} > {}".format(message, dropped, max_dropped))
        return None, None, None, dropped, [-1, -1]

    # shape_before = x.shape[1]
    # sel = VarianceThreshold(threshold=0)
    # x = sel.fit_transform(x)
    # shape_after = x.shape[1]
    n_variables = len(_BANDS)
    n_observations, m_temp = x.shape
    n_features = m_temp // n_variables
    print("initial matrix shape:", x.shape, "(n_observations: %d, n_features: %d, n_variables: %d)" % (n_observations,
                                                                                                       n_features,
                                                                                                       n_variables))
    print("target number of features per band: ", k, "total target features:", k * len(_BANDS))
    # print(shape_before, shape_after)

    ######################################################
    # pipeline for multivariate application
    #######################################################

    # get log-tf-idf
    vsm = VSM(class_based=False, classes=classes, norm=lsa_kwargs["normalize"], use_idf=lsa_kwargs["use_idf"],
              smooth_idf=True, sublinear_tf=lsa_kwargs["sublinear_tf"])

    # feature selection to k (gives matrix of shape (m, k, b))
    manova = GeneralSelectKTop(k, manova_rank, allow_nd=True, n_variables=n_variables)

    # normalize the resulting features
    normalize = Normalizer()

    # scaler
    scaler = StandardScaler()

    # latent semantic analysis
    lsa = LSA(sc=k//2, algorithm="randomized", n_iter=5, random_state=None, tol=0.)

    # centroid prototype
    centroid = CentroidClass(classes=classes)

    # flattening the matrix (gives matrix of shape (m, k*b))
    # flattening = Flatten3Dto2D()

    # KNN classifier with KD_Tree and custom distance (dependent DTW for multivariate application)
    # knn = KNeighborsClassifier(n_neighbors=1, leaf_size=10, metric=flattened_ddtw, metric_params={"shape": (k, n_variables)})
    knn = KNeighborsClassifier(n_neighbors=1)
    knn2 = knnclassifier(classes=classes, useClasses=True, metric=flattened_ddtw, metric_params={"shape": (k, n_variables)})

    pipeline = Pipeline([
        ("manova", manova),
        ("vsm", vsm),
        # ("lsa", lsa),
        # ("flattening", flattening),
        # ("normalize", normalize),
        # ("scaler", scaler),
        ("centroid", centroid),
        ("knn", knn2),
        # ("knn", knn)
    ])
    print("[%s]: DOING CROSS VALIDATION..." % message, end="\r")
    ini = time.time()
    scores = cross_val_score(pipeline, x, labels, scoring="balanced_accuracy", cv=cv, n_jobs=_N_JOBS, verbose=0)
    end = time.time()
    print("[%s]: %.3f += %.3f (time: %.3f sec)" % (message, float(np.mean(scores)), float(np.std(scores)), end - ini))

    return scores, pipeline, None, dropped, [-1,-1]


def save_first_stack(x_score, win_list, wl_list, base_file_name, out_path):
    out_file = os.path.join(out_path, base_file_name + "1-top_0-" + time.strftime(
        "%Y%m%d-%H%M%S") + ".csv")
    f = open(out_file, "a+")
    header = "base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
    f.write(header)
    f.close()

    msg = "[{win,wl}]"
    for i in range(len(win_list)):
        win = win_list[i]
        wl = wl_list[i]
        scores, pipeline, dropped, shapes = x_score[win][wl]
        # line to write to file
        line = "%s,%d,%f,%d,%d,%d," % (msg, wl, win, dropped, shapes[0], shapes[1])

        if scores is None and pipeline is None:
            # raise an error
            raise ValueError("error")
        # else
        else:
            # generate the rest of the line to write
            if "lsa" in pipeline:
                exp_var  = np.sum(pipeline["lsa"].explained_variance_ratio_)
                n_comps = pipeline["lsa"].n_components
            elif "manova" in pipeline:
                exp_var = -1
                n_comps = pipeline["manova"].k
            else:
                exp_var = -1
                n_comps = -1

            scheme_name = pipeline["vsm"].get_scheme_notation()
            mean_cv = float(np.mean(scores))
            std_cv = float(np.std(scores))
            line += "%f,%f,%f,%d,%s\n" % (mean_cv, std_cv,
                                          exp_var,
                                          n_comps,
                                          pipeline["vsm"].get_scheme_notation())
        f = open(out_file, "a+")
        try:
            f.write(line)
        except:
            f.close()
        else:
            f.close()


def general_grid_search(data, labels, wins, wls, alphabet_size, q, classes, lsa_kwargs, doc_kwargs, out_path,
                        k=100, max_dropped="default", cv=5):

    # quantities reduced name
    q_code = quantities_code(q)

    # base name for outpu files
    base_file_name = "multi_ress-%s-stack_" % q_code

    # compute repr for all options
    x_repr, x_score, x_acc_list, win_list, wl_list = get_all_single_representations(data, labels, wins, wls,
                                                                                    alphabet_size, q, classes,
                                                                                    lsa_kwargs, doc_kwargs, k=k,
                                                                                    max_dropped=max_dropped,
                                                                                    cv=cv)
    # save results of first iteration
    save_first_stack(x_score, win_list, wl_list, base_file_name, out_path)

    # rank results by acc
    rank_by_idx = np.argsort(x_acc_list)[::-1]

    # test on top 5 results
    for ii, top_idx in enumerate(rank_by_idx[:5]):
        # incremental config
        x = x_repr[win_list[top_idx]][wl_list[top_idx]]
        record_idxs = [top_idx]
        record_acc = x_acc_list[top_idx]
        print("---> starting stack of configurations for top ", ii+1)
        msg = "[{win,wl}"
        for used_idx in record_idxs:
            msg += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
        msg += "]"
        print("---> best config stack 1: %s ::::: acc: %.3f" % (msg, float(record_acc)))
        # stack up until 5 configs
        for jj in [2, 3, 4, 5]:
            try:
                print("---> starting search on stack level ", jj)
                # define an output file
                out_file = os.path.join(out_path, base_file_name + str(jj) + "-top_" + str(ii+1) + "-" + time.strftime("%Y%m%d-%H%M%S") + ".csv")
                f = open(out_file, "a+")
                header = "base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
                f.write(header)
                f.close()

                # define iter parameters to find best config
                best_config = None
                best_acc = record_acc

                # start iter to find next stack config
                for next_idx in rank_by_idx:
                    # if next_idx was not added to incremental config
                    if next_idx not in record_idxs:
                        # try to stack this idx and check acc
                        try:
                            # generate a compy of the repr matrix
                            x_i = sparse.hstack([x, x_repr[win_list[next_idx]][wl_list[next_idx]]], format="csr")

                            # generate a custom message
                            message = "{win,wl}"
                            for used_idx in record_idxs:
                                message += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
                            message += ",{%.3f,%d}" % (win_list[next_idx], wl_list[next_idx])

                            # get a cr score from this current stack of configs
                            score, pipeline, _, dropped, shapes = general_cv_score(x_i, labels, classes, lsa_kwargs,
                                                                                   k=k, max_dropped=max_dropped,
                                                                                   message=message, cv=cv)
                            # line to write to file
                            line = "%s,%d,%f,%d,%d,%d," % (message,wl_list[next_idx], win_list[next_idx], dropped, shapes[0], shapes[1])

                            # if the iteration  was invalid (should always be valid here)
                            if score is None and pipeline is None:
                                # raise an error
                                raise ValueError("error")
                            # else
                            else:
                                # generate the rest of the line to write
                                exp_var = np.sum(pipeline["lsa"].explained_variance_ratio_)
                                n_comps = pipeline["lsa"].n_components
                                scheme_name = pipeline["vsm"].get_scheme_notation()
                                mean_cv = float(np.mean(score))
                                std_cv = float(np.std(score))
                                line += "%f,%f,%f,%d,%s\n" % (mean_cv, std_cv,
                                                      exp_var if exp_var is not None else -1,
                                                      pipeline["lsa"].n_components,
                                                      pipeline["vsm"].get_scheme_notation())

                                # and if this stack is the best so far, update variables
                                if mean_cv > best_acc:
                                    best_config = record_idxs + [next_idx]
                                    best_acc = mean_cv
                        except Exception as e:
                            print("failed on stack step k=%d, for wl=%d, win=%f, error: %s" % (jj, wl_list[next_idx], win_list[next_idx], e))
                        else:

                            # write line to file
                            f = open(out_file, "a+")
                            try:
                                f.write(line)
                            except:
                                f.close()
                            else:
                                f.close()

                # after the iteration, if no improvement was achieve, stop the process
                if best_config is None or best_acc <= record_acc:
                    print("ACCURACY IS NOT IMPROVING, STOPPING CODE")
                    break
                else:
                    # update the incremental configuration and advance to next stack level
                    x_arr = []
                    for used_idx in best_config:
                        x_arr.append(x_repr[win_list[used_idx]][wl_list[used_idx]])
                    x = sparse.hstack(x_arr, format="csr")
                    record_idxs = best_config
                    record_acc = best_acc
                    msg = "[{win,wl}"
                    for used_idx in record_idxs:
                        msg += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
                    msg += "]"
                    print("---> best config stack %d: %s ::::: acc: %.3f" % (k, msg, float(record_acc)))
            except Exception as e:
                print("failed stack step k=%d, error: %s" % (k, e))


if __name__ == '__main__':
    # inputs

    set_name = "plasticc_train"

    # read dataset
    dataset, labels_, metadata, split_folds = gen_dataset_from_h5(set_name, num_folds=5)
    split_folds = rearrange_splits(split_folds)
    classes = np.unique(labels_)
    # pdb.set_trace()
    # estimate spatial complexity
    sc = int(np.mean([len(ts.observations["flux"]) * 2 for ts in dataset]))

    # estimate max window of observation
    time_durations = np.array(
        [ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    max_window = mean_time + std_time

    # define some fixed parameters
    wls = [1, 2]
    wins = np.logspace(np.log10(10), np.log10(mean_time + std_time), 30)

    doc_kwargs = {
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }

    lsa_kwargs = {
        "class_based": True, # options: True, False
        "normalize": "l2", # options: None, l2
        "use_idf": True, # options: True, False
        "sublinear_tf": True # options: True, False
    }

    alphabet_size = 6
    q = [["mean", "trend", "min_max"]]

    # out path
    out_path = os.path.join("..", "data", "configs_results", "manova_multi_ress_search")
    if not os.path.exists(out_path):
        raise ValueError("folder doesnt exists")

    general_grid_search(dataset, labels_, wins, wls, alphabet_size, q, classes, lsa_kwargs, doc_kwargs, out_path,
                        k=sc//len(_BANDS), cv=split_folds)


