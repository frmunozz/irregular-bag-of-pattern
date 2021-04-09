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


## fixed parameters
_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
_N_JOBS = 8

symbols = {
    "mean": "M",
    "std": "S",
    "trend": "T",
    "min_max": "m",
}


def cv_score(data, labels, win, wl, alphabet_size, q, classes, lsa_kwargs, doc_kwargs,
             spatial_comp=100, max_dropped="default"):

    ini = time.time()
    if len(q) > 1:
        x_arr = []
        for q_i in q:
            doc_kwargs["quantity"] = np.array(q_i)
            doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q_i))
            text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                       opt_desc=", " + "-".join(q_i), **doc_kwargs)
            x_i = text_gen.fit_transform(data)
            x_arr.append(x_i)
        x = sparse.hstack(x_arr, format="csr")
    else:
        doc_kwargs["quantity"] = np.array(q[0])
        doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q[0]))
        text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                   opt_desc=", " + "-".join(q[0]), **doc_kwargs)
        x = text_gen.fit_transform(data)

    if max_dropped == "default":
        max_dropped = int(0.05 * len(labels))
    dropped = len(np.where(np.sum(x, axis=1) == 0)[0])
    if dropped > max_dropped:
        print("[wl:{}, win: {}] dropped because {} > {}".format(wl, win, dropped, max_dropped))
        return None, None, None, dropped, [-1, -1]

    shape_before = x.shape[1]
    sel = VarianceThreshold(threshold=0)
    x = sel.fit_transform(x)
    shape_after = x.shape[1]
    # print(shape_before, shape_after)

    # define pipeline for CV score
    vsm = VSM(class_based=False, classes=classes, norm=lsa_kwargs["normalize"], use_idf=lsa_kwargs["use_idf"],
              smooth_idf=True, sublinear_tf=lsa_kwargs["sublinear_tf"])
    lsa = LSA(sc=min(spatial_comp, x.shape[1] - 1), algorithm="randomized", n_iter=5, random_state=None, tol=0.)
    centroid = CentroidClass(classes=classes)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=lsa_kwargs["class_based"])
    pipeline = Pipeline([("vsm", vsm), ("lsa", lsa), ("centroid", centroid), ("knn", knn)])
    scores = cross_val_score(pipeline, x, labels, scoring="balanced_accuracy", cv=5, n_jobs=_N_JOBS, verbose=0)
    end = time.time()
    print("[win: %.3f, wl: %d]:" % (win, wl), np.mean(scores), "+-", np.std(scores), " (time: %.3f sec)" % (end - ini))
    return scores, pipeline, None, dropped, [shape_before, shape_after]


def grid_search(data, labels, wins, wls, alphabet_size, q, classes, lsa_kwargs, doc_kwargs, out_path, spatial_comp=100):
    # make file to write results
    f_name = ""
    for q_i in q:
        for q_ii in q_i:
            f_name += symbols[q_ii]
        f_name += "-"
    f_name += "single_res_results-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    print("USING FILE: ", f_name, ", for quantities: ", q)
    header = "wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
    f = open(os.path.join(out_path, f_name), "a+")
    f.write(header)
    f.close()

    for wl in wls:
        for win in wins:
            try:
                scores, pipeline, text_gen, dropped, shapes = cv_score(data, labels, win, wl, alphabet_size,
                                                               q, classes, lsa_kwargs, doc_kwargs,
                                                                       spatial_comp=spatial_comp)
                # print("exp var: ", np.sum(pipeline["lsa"].explained_variance_ratio_))
                print("wl: ", wl, ", win: ", win, ", dropped: ", dropped, ", shapes: ", shapes)
                line = "%d,%f,%d,%d,%d," % (wl, win, dropped, shapes[0], shapes[1])

                if scores is None and pipeline is None:
                    line += "%f,%f,%f,%d,%s\n" % (0, 0, 0, 0, "nnn")
                else:
                    exp_var = np.sum(pipeline["lsa"].explained_variance_ratio_)
                    n_comps = pipeline["lsa"].n_components
                    scheme_name = pipeline["vsm"].get_scheme_notation()
                    mean_cv = np.mean(scores)
                    std_cv = np.std(scores)
                    print("n_comps:", n_comps, ", scheme name: ", scheme_name, ", exp_var: ", exp_var, ", mean_cv: ", mean_cv, ", std_cv: ", std_cv)
                    line += "%f,%f,%f,%d,%s\n" % (mean_cv, std_cv,
                                              exp_var if exp_var is not None else -1,
                                              pipeline["lsa"].n_components,
                                              pipeline["vsm"].get_scheme_notation())
            except Exception as e:
                print("failed interation wl=%d, win=%f, error: %s" % (wl, win, e))
            else:
                f = open(os.path.join(out_path, f_name), "a+")
                try:
                    f.write(line)
                except:
                    f.close()
                else:
                    f.close()


if __name__ == '__main__':
    # inputs

    set_name = "plasticc_train"

    # read dataset
    dataset, labels_, metadata = gen_dataset_from_h5(set_name)
    classes = np.unique(labels_)

    # estimate spatial complexity
    sc = int(np.mean([len(ts.observations["flux"]) * 2 for ts in dataset]))

    # estimate max window of observation
    time_durations = np.array(
        [ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    max_window = mean_time + std_time

    # define some fixed parameters
    wls = [1, 2, 3, 4, 5]
    wins = np.logspace(np.log10(10), np.log10(mean_time + std_time), 20)

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

    alphabet_size = 4

    # iter over different configurations
    # Sm, Mm, MS, MT, Tm, ST
    q_arr = [
        [["mean", "std"], ["mean", "trend"], ["std", "min_max"]],  # MS-MT-Sm
        [["mean", "std"], ["mean", "trend"], ["mean", "min_max"]],  # MS-MT-Mm
        [["mean", "std"], ["mean", "trend"], ["trend", "min_max"]],  # MS-MT-Tm**
        [["mean", "std"], ["mean", "trend"], ["std", "trend"]],  # MS-MT-ST
        [["mean", "std"], ["trend", "min_max"], ["std", "min_max"]],  # MS-Tm-Sm
        # [["mean", "std"], ["trend", "min_max"], ["mean", "min_max"]],  # MS-Tm-Mm **
        [["mean", "std"], ["trend", "min_max"], ["mean", "trend"]],  # MS-Tm-MT
        [["mean", "std"], ["trend", "min_max"], ["std", "trend"]],  # MS-Tm-ST
        [["mean", "trend"], ["mean", "min_max"], ["std", "min_max"]],  # MT-Mm-Sm
        # [["mean", "trend"], ["mean", "min_max"], ["mean", "std"]],  # MT-Mm-MS**
        [["mean", "trend"], ["mean", "min_max"], ["trend", "min_max"]],  # MT-Mm-Tm
        [["mean", "trend"], ["mean", "min_max"], ["std", "trend"]],  # MT-Mm-ST
    ]

    # out path
    out_path = os.path.join("..", "data", "configs_results")

    for q in q_arr:
        grid_search(dataset, labels_, wins, wls, alphabet_size, q, classes, lsa_kwargs, doc_kwargs, out_path,
                    spatial_comp=sc)


