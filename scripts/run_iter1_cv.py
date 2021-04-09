import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import pandas as pd
import time
from scipy import sparse

from src.preprocesing import gen_dataset, gen_dataset_from_h5
from src.feature_extraction.text import ParameterSelector, MPTextGenerator, TextGeneration, CountVectorizer
from src.feature_extraction.vector_space_model import VSM
from src.feature_extraction.centroid import CentroidClass
from src.feature_selection.select_k_best import SelectKTop
from src.decomposition import LSA, PCA
from src.neighbors import KNeighborsClassifier
from src.feature_extraction.window_slider import TwoWaysSlider

from sklearn.feature_selection import VarianceThreshold

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
_N_JOBS = 8


def cv_score(d_train, l_train, win, wl, doc_kwargs,
             class_based=False, classes=None, normalize='l2',
             use_idf=True, sublinear_tf=True, max_dropped="default", sc=100, merged_quantities=True):
    ini = time.time()
    if len(doc_kwargs["quantity"]) > 1 and not merged_quantities:
        quantities = doc_kwargs["quantity"]
        alph_sizes = doc_kwargs["alphabet_size"]
        x_arr = []
        for q, a_size in zip(quantities, alph_sizes):
            doc_kwargs["quantity"] = np.array([q])
            doc_kwargs["alphabet_size"] = np.array([a_size])
            text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                       opt_desc=", %s" % q, **doc_kwargs)
            x_i = text_gen.fit_transform(d_train)
            x_arr.append(x_i)
        x = sparse.hstack(x_arr, format="csr")
        doc_kwargs["quantity"] = quantities
        doc_kwargs["alphabet_size"] = alph_sizes

    else:
        text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2, **doc_kwargs)
        x = text_gen.fit_transform(d_train)

    if max_dropped == "default":
        max_dropped = int(0.05 * len(l_train))
    dropped = len(np.where(np.sum(x, axis=1) == 0)[0])
    if dropped > max_dropped:
        print("dropped because {} > {}".format(dropped, max_dropped))
        return None, None, None, dropped

    shape_before = x.shape
    sel = VarianceThreshold(threshold=0)
    x = sel.fit_transform(x)

    print("shape before variance drop:", shape_before, ", shape after:", x.shape)

    # define pipeline for CV score
    vsm = VSM(class_based=False, classes=classes, norm=normalize, use_idf=use_idf,
              smooth_idf=True, sublinear_tf=sublinear_tf)
    lsa = LSA(sc=min(sc, x.shape[1]-1), algorithm="randomized", n_iter=5, random_state=None, tol=0.)
    centroid = CentroidClass(classes=classes)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipeline = Pipeline([("vsm", vsm), ("lsa", lsa), ("centroid", centroid), ("knn", knn)])
    scores = cross_val_score(pipeline, x, l_train, scoring="balanced_accuracy", cv=5, n_jobs=_N_JOBS, verbose=0)
    end = time.time()
    print("[win: %.3f, wl: %d]:" % (win, wl), np.mean(scores), "+-", np.std(scores), " (time: %.3f sec)" % (end-ini))
    return scores, pipeline, None, dropped


def cv_score_multi_res(x, l_train, text="X",
                       class_based=False, classes=None, normalize='l2',
                       use_idf=True, sublinear_tf=True, max_dropped="default", sc=100):
    ini = time.time()

    if max_dropped == "default":
        max_dropped = int(0.05 * len(l_train))
    dropped = len(np.where(np.sum(x, axis=1) == 0)[0])
    if dropped > max_dropped:
        return None, None, dropped
    vsm = VSM(class_based=False, classes=classes, norm=normalize, use_idf=use_idf,
              smooth_idf=True, sublinear_tf=sublinear_tf)
    lsa = LSA(sc=min(sc, x.shape[1]-1), algorithm="randomized", n_iter=5, random_state=None, tol=0.)
    centroid = CentroidClass(classes=classes)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipeline = Pipeline([("vsm", vsm), ("lsa", lsa), ("centroid", centroid), ("knn", knn)])
    scores = cross_val_score(pipeline, x, l_train, scoring="balanced_accuracy", cv=5, n_jobs=_N_JOBS, verbose=0)
    end = time.time()
    print("[%s]:" % text, np.mean(scores), "+-", np.std(scores), " (time: %.3f sec)" % (end-ini))
    return scores, pipeline, dropped


def grid_search(d_train, l_train, wins, wls, doc_kwargs, class_based=False, classes=None, normalize='l2',
                use_idf=True, sublinear_tf=True, spatial_comp=100, merged_quantities=True):

    output_dict = defaultdict(list)
    for wl in wls:
        for win in wins:
            scores, pipeline, text_gen, dropped = cv_score(d_train, l_train, win, wl, doc_kwargs,
                                                           class_based=class_based, classes=classes,
                                                           normalize=normalize, use_idf=use_idf,
                                                           sublinear_tf=sublinear_tf, sc=spatial_comp,
                                                           merged_quantities=merged_quantities)
            if scores is None and pipeline is None:
                # the iteration failed and we skip
                output_dict["wl"].append(wl)
                output_dict["win"].append(win)
                output_dict["dropped"].append(dropped)
                output_dict["mean_cv"].append(0)
                output_dict["std_cv"].append(0)
                output_dict["exp_var"].append(0)
                output_dict["n_comp"].append(0)
                output_dict["scheme"].append("nnn")
            else:
                # otherwise, we add the results to output dict
                output_dict["wl"].append(wl)
                output_dict["win"].append(win)
                output_dict["dropped"].append(dropped)
                output_dict["mean_cv"].append(np.mean(scores))
                output_dict["std_cv"].append(np.std(scores))
                output_dict["exp_var"].append(np.sum(pipeline["lsa"].explained_variance_ratio_))
                output_dict["n_comp"].append(pipeline["lsa"].n_components)
                output_dict["scheme"].append(pipeline["vsm"].get_scheme_notation())

    return output_dict


def grid_search_multi_rest(d_train, l_train, wins, wls, accs, doc_kwargs, class_based=False, classes=None, normalize='l2',
                use_idf=True, sublinear_tf=True, spatial_comp=100, merged_quantities=True):
    limit = 4
    tops = 5
    x_arr = []
    for win, wl in zip(wins, wls):
        text_gen_i = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2, **doc_kwargs)
        x_i = text_gen_i.fit_transform(d_train)
        shape_before = x_i.shape
        sel = VarianceThreshold(threshold=0)
        x_i = sel.fit_transform(x_i)
        print("shape before variance drop:", shape_before, ", shape after:", x_i.shape)
        x_arr.append(x_i)

    idxs = np.arange(len(wls))

    output_dict = defaultdict(list)
    # try for top 5 params
    for i in range(tops):
        idx_sel = [i]
        try_idxs = np.delete(idxs, i)

        global_best_score = accs[i]
        print("starting combination: ", idx_sel, " ".join(["(%.3f, %d)" % (wins[_k], wls[_k]) for _k in idx_sel]))
        print("starting accuracy:", global_best_score)
        global_scores = None
        global_pipeline = None
        global_dropped = None
        while len(idx_sel) < limit:
            local_best_score = -1
            local_scores = None
            local_pipeline = None
            local_dropped = None
            local_j = None
            for j in try_idxs:
                x_sel = sparse.hstack([x_arr[q] for q in idx_sel + [j]], format="csr")
                text = " ".join(["(%.3f, %d)" % (wins[_k], wls[_k]) for _k in idx_sel + [j]])
                scores, pipeline, dropped = cv_score_multi_res(x_sel, l_train, text=text,
                                                               class_based=class_based, classes=classes,
                                                               normalize=normalize, use_idf=use_idf,
                                                               sublinear_tf=sublinear_tf, sc=spatial_comp)
                if scores is None and pipeline is None:
                    # this should never happen
                    raise ValueError("this is wrong")

                mean_score = np.mean(scores)
                if mean_score > local_best_score:
                    local_best_score = mean_score
                    local_dropped = dropped
                    local_pipeline = pipeline
                    local_scores = scores
                    local_j = j

            if local_best_score >= global_best_score:
                print("local best improve, iterate again")
                if len(idx_sel) >= limit:
                    print("reach max number of iter, break")
                    break
                idx_sel = idx_sel + [local_j]

                try_idxs_pos = np.where(try_idxs == local_j)[0][0]
                try_idxs = np.delete(try_idxs, try_idxs_pos)

                global_scores = local_scores
                global_best_score = local_best_score
                global_dropped = local_dropped
                global_pipeline = local_pipeline

                print("current combinations:", idx_sel,
                      " ".join(["(%.3f, %d)" % (wins[_k], wls[_k]) for _k in idx_sel]))
                print("current best acc: ", global_best_score)
                print("options to kee trying:", try_idxs)

            else:
                print("local best doesnt improve, break")
                print("local best:", local_best_score, ", global best:", global_best_score)
                break
        print("final best:", idx_sel, " ".join(["(%.3f, %d)" % (wins[_k], wls[_k]) for _k in idx_sel]))
        print("final best score:", global_best_score)

        output_dict["pivot_win"].append(wins[i])
        output_dict["pivot_wl"].append(wls[i])
        output_dict["wins"].append(" ".join(["%.3f" % wins[_k] for _k in idx_sel]))
        output_dict["wls"].append(" ".join(["%d" % wls[_k] for _k in idx_sel]))
        if global_dropped is None:
            output_dict["dropped"].append(0)
            output_dict["mean_cv"].append(0)
            output_dict["std_cv"].append(0)
            output_dict["exp_var"].append(0)
            output_dict["n_comp"].append(0)
            output_dict["scheme"].append("nnn")
        else:
            output_dict["dropped"].append(global_dropped)
            output_dict["mean_cv"].append(np.mean(global_scores))
            output_dict["std_cv"].append(np.std(global_scores))
            output_dict["exp_var"].append(np.sum(global_pipeline["lsa"].explained_variance_ratio_))
            output_dict["n_comp"].append(global_pipeline["lsa"].n_components)
            output_dict["scheme"].append(global_pipeline["vsm"].get_scheme_notation())

    return output_dict


if __name__ == '__main__':
    # set_name = "plasticc_augment_ddf_100"
    set_name = "plasticc_train"

    ini = time.time()
    dataset, labels_, metadata = gen_dataset_from_h5(set_name)
    classes = np.unique(labels_)
    sc = int(np.mean([len(ts.observations["flux"]) * 2 for ts in dataset]))
    print("dataset mean spatial complexity:", sc)
    time_durations = np.array(
        [ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)

    wls = [1, 2, 3, 4]
    wins = np.logspace(np.log10(10), np.log10(mean_time + std_time), 20)
    # wins = [50, 80]
    print("windows:", wins)
    # sc = 200

    doc_kwargs = {
        "alphabet_size": np.array([4, 4, 4]),
        "quantity": np.array(["mean", "trend", "std"]),
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }
    merged = True
    class_based = True  # options: True, False
    normalize = 'l2'  # options: None, l2
    use_idf = True  # options: True, False
    sublinear_tf = True  # options: True, False

    print("::::::::::::::: START GRID SEARCH CV ::::::::::::::: ")
    output_dict = grid_search(dataset, labels_, wins, wls, doc_kwargs,
                              class_based=class_based, classes=classes,
                              normalize=normalize, use_idf=use_idf,
                              sublinear_tf=sublinear_tf, spatial_comp=sc, merged_quantities=merged)
    print("::::::::::::::: END GRID SEARCH CV ::::::::::::::: ")
    print(":::::::::::::::::::::::::::::::::::::::::::::::::: ")
    print("::::::::::::::: SAVING ::::::::::::::::::::::::::: ")
    df = pd.DataFrame(output_dict)
    out_file = os.path.join(main_path, "data", "results", "plasticc", "iter1_%s_mean-trend-std-4.csv" % set_name)
    df.to_csv(out_file, index=False)

    # df = pd.read_csv(out_file)
    # df = df[df["scheme"] != "nnn"]
    # df = df.sort_values("mean_cv", ascending=False)
    # iter2_wls = df["wl"].to_numpy()
    # iter2_wins = df["win"].to_numpy()
    # iter2_accs = df["mean_cv"].to_numpy()
    # print(":::::::::::::::::::::::::::::::::::::::::::::::::: ")
    # print("::::::::::::::: START GRID SEARCH CV 2 ::::::::::::::: ")
    # output_dict2 = grid_search_multi_rest(dataset, labels_, iter2_wins, iter2_wls, iter2_accs, doc_kwargs,
    #                                       class_based=class_based, classes=classes,
    #                                       normalize=normalize, use_idf=use_idf,
    #                                       sublinear_tf=sublinear_tf, spatial_comp=sc)
    # print("::::::::::::::: END GRID SEARCH CV 2 ::::::::::::::: ")
    # print(":::::::::::::::::::::::::::::::::::::::::::::::::: ")
    # print("::::::::::::::: SAVING ::::::::::::::::::::::::::: ")
    # df = pd.DataFrame(output_dict2)
    # out_file = os.path.join(main_path, "data", "results", "plasticc", "iter2_%s_mean_trend.csv" % set_name)
    # df.to_csv(out_file, index=False)
    print("::::::::: TOTAL RUN TIME %.3f :::::::::::::::::::: " % (time.time() - ini))


