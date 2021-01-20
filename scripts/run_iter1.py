import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)

from src.preprocesing import gen_dataset, gen_dataset_from_h5
from src.pipelines import PipelineBuilder
from src.feature_extraction.text import MPTextGenerator, CountVectorizer


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import sparse
import time
import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV

from src.feature_extraction.text import ParameterSelector, MPTextGenerator, TextGeneration, CountVectorizer
from src.feature_extraction.vector_space_model import VSM
from src.feature_extraction.centroid import CentroidClass
from src.feature_selection.select_k_best import SelectKTop
from src.decomposition import LSA, PCA
from src.neighbors import KNeighborsClassifier
from src.feature_extraction.window_slider import TwoWaysSlider

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, VarianceThreshold


def first_iter_pipeline(win, wl, doc_kwargs, class_based=False, classes=None, normalize='l2',
                        use_idf=True, sublinear_tf=True):

    pipe = []

    # extract the word features from dataset as a bag-of-word
    text_gen = TextGeneration(win=win, wl=wl, direct_bow=True, **doc_kwargs)
    pipe.append(("textgen", text_gen))

    # transform the words to a basic VSM called bag-of-word
    # count_vec = CountVectorizer(alph_size=text_gen.alph_size,
    #                             irr_handler=text_gen.irr_handler,
    #                             word_length=text_gen.wl)
    # pipe.append(("count_vec", count_vec))

    # generate VSM following a fixed scheme
    vsm = VSM(class_based=False, classes=classes, norm=normalize, use_idf=use_idf,
              smooth_idf=True, sublinear_tf=sublinear_tf)
    pipe.append(("vsm", vsm))

    # we will do it without LSA

    # if class based, set centroid
    if class_based:
        centroid = CentroidClass(classes=classes)
        pipe.append(("prototype", centroid))

    # set classifier as 1-NN
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipe.append(("knn", knn))

    pipeline = Pipeline(pipe)
    return pipeline


def second_iter_pipeline(win, wl, doc_kwargs, class_based=False, classes=None, normalize='l2',
                        use_idf=True, sublinear_tf=True):

    pipe = []

    # extract the word features from dataset as a bag-of-word
    text_gen = TextGeneration(win=win, wl=wl, direct_bow=True, **doc_kwargs)
    pipe.append(("textgen", text_gen))

    # transform the words to a basic VSM called bag-of-word
    # count_vec = CountVectorizer(alph_size=text_gen.alph_size,
    #                             irr_handler=text_gen.irr_handler,
    #                             word_length=text_gen.wl)
    # pipe.append(("count_vec", count_vec))

    # generate VSM following a fixed scheme
    vsm = VSM(class_based=False, classes=classes, norm=normalize, use_idf=use_idf,
              smooth_idf=True, sublinear_tf=sublinear_tf)
    pipe.append(("vsm", vsm))

    # use LSA with fixed number of components
    lsa = LSA(sc=149, algorithm="randomized", n_iter=5, random_state=None, tol=0.)
    pipe.append(("red", lsa))

    # if class based, set centroid
    if class_based:
        centroid = CentroidClass(classes=classes)
        pipe.append(("prototype", centroid))

    # set classifier as 1-NN
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipe.append(("knn", knn))

    pipeline = Pipeline(pipe)
    return pipeline


merged_labels_to_num = {
    "Single microlens": 1,
    "TDE": 2,
    "Short period VS": 3,
    "SN": 4,
    "M-dwarf": 5,
    "AGN": 6,
    "Unknown": 99
}

merged_labels = {
    6: "Single microlens",
    15: "TDE",
    16: "Short period VS",
    42: "SN",
    52: "SN",
    53: "Short period VS",
    62: "SN",
    64: "SN",
    65: "M-dwarf",
    67: "SN",
    88: "AGN",
    90: "SN",
    92: "Short period VS",
    95: "SN",
    99: "Unknown"
}


if __name__ == '__main__':
    doc_kwargs = {
        "alphabet_size": np.array([4]),
        "quantity": np.array(["mean"]),
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": False
    }
    class_based = True  # options: True, False
    normalize = 'l2'  # options: None, l2
    use_idf = True  # options: True, False
    sublinear_tf = True  # options: True, False

    print("loading dataset")
    dataset, labels_, metadata = gen_dataset_from_h5("plasticc_augment_ddf_50")
    print("done")
    bands = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    # labels = np.array([merged_labels_to_num[merged_labels[x]] for x in labels_])
    labels = labels_
    classes = np.unique(labels)

    ##########################################################
    ## GRID SEARCH USING CROSS VALIDATION K FOLD
    ##########################################################

    # pipeline = first_iter_pipeline(80, 6, doc_kwargs, class_based=class_based, classes=classes, normalize=normalize,
    #                                use_idf=use_idf, sublinear_tf=sublinear_tf)
    pipeline = second_iter_pipeline(80, 6, doc_kwargs, class_based=class_based, classes=classes, normalize=normalize,
                                   use_idf=use_idf, sublinear_tf=sublinear_tf)

    print("simple cross val on wl 5 win 80")
    print(cross_val_score(pipeline, dataset, labels, scoring="balanced_accuracy", verbose=2, n_jobs=5))
    print("-------------------------------")
    wls = np.array([2, 3, 4, 5, 6])
    wins = np.array([10])  #days
    scs = np.array([100, 200, 400, 600])
    parameters = {"textgen__wl": wls, "textgen__win": wins}
    print("starting grid search")
    grid_search = GridSearchCV(pipeline, parameters, verbose=2, cv=5, n_jobs=6, scoring="balanced_accuracy")
    t0 = time.time()
    grid_search.fit(dataset, labels)
    print("DONE IN %0.3fs" % (time.time() - t0))
    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    ############################################################
    ## SAVING GRID SEARCH RESULTS TO CSV FILE
    ###########################################################
    c = "prototype" if class_based else None

    scheme = "l" if sublinear_tf else "n"
    scheme += "t" if use_idf else "n"
    scheme += "c" if normalize == "l2" else "n"
    scheme += "." + scheme

    n = len(grid_search.cv_results_["params"])

    data_dict = {
        "class_feature": np.full(n, c),
        "scheme": np.full(n, scheme),
        # "smooth_idf": np.full(n, smooth_idf),
        "reducer_type": np.full(n, 'LSA'),
        # "n_neighbors": np.full(n, n_neighbors),
    }
    for k in grid_search.param_grid.keys():
        data_dict["param_" + k] = grid_search.cv_results_["param_" + k]
    data_dict["mean_test_score"] = grid_search.cv_results_["mean_test_score"]
    data_dict["std_test_score"] = grid_search.cv_results_["std_test_score"]
    data_dict["rank_test_score"] = grid_search.cv_results_["rank_test_score"]
    data_dict["variance"] = [grid_search.estimator["red"]]

