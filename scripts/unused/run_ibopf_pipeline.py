import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)

from src.preprocesing import gen_dataset_from_h5
from src.pipelines import PipelineBuilder
from src.feature_extraction.text import MPTextGenerator, CountVectorizer


import numpy as np
from scipy import sparse
import time
import pandas as pd

from sklearn.model_selection import GridSearchCV


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


def pipeline_grid_search(classes, values, out_wins, out_wls, spatial_comp, X, labels_merged,
                         class_based, class_type, norm_key, use_idf,
                         smooth_idf, sublinear_tf, reducer_type, n_neighbors=1, n_iter=20):
    c = class_type if class_based else None

    scheme = "l" if sublinear_tf else "n"
    scheme += "t" if use_idf else "n"
    scheme += "c" if norm_key == "l2" else "n"
    scheme += "." + scheme

    print("#################################################")
    print("#################################################")
    print("TESTING GRID SEARCH FOR PIPELINE:")
    print("---> CLASS FEATURE: ", c)
    print("---> SCHEME: ", scheme)
    print("---> REDUCER TYPE: ", reducer_type)
    ##########################################
    ## BUILD PIPELINE
    #########################################

    pipeline_builder = PipelineBuilder(class_based=class_based, class_type=class_type, classes=classes)
    pipeline_builder.set_feature_extraction(precomputed=True, idx=-1, data=values, win_arr=out_wins, wl_arr=out_wls)
    pipeline_builder.set_transformer(norm=norm_key, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    pipeline_builder.set_reducer(spatial_complexity=spatial_comp, reducer_type=reducer_type, n_iter=n_iter)
    pipeline_builder.set_normalizer()
    pipeline_builder.set_classifier(n_neighbors=n_neighbors)
    pipeline = pipeline_builder.build()

    ##########################################################
    ## GRID SEARCH USING CROSS VALIDATION K FOLD
    ##########################################################

    parameters = {
        "ext__idx": np.arange(len(values)),
        "red__sc": np.linspace(10, spatial_comp - 1, 10, dtype=int)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=1, cv=10)
    t0 = time.time()
    grid_search.fit(X, labels_merged)

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
    n = len(grid_search.cv_results_["params"])

    data_dict = {
        "class_feature": np.full(n, c),
        "scheme": np.full(n, scheme),
        # "smooth_idf": np.full(n, smooth_idf),
        "reducer_type": np.full(n, reducer_type),
        # "n_neighbors": np.full(n, n_neighbors),
    }
    for k in grid_search.param_grid.keys():
        data_dict["param_" + k] = grid_search.cv_results_["param_" + k]

    data_dict["window"] = np.array([out_wins[k] for k in data_dict["param_ext__idx"]])
    data_dict["word_length"] = np.array([out_wls[k] for k in data_dict["param_ext__idx"]])
    data_dict["mean_test_score"] = grid_search.cv_results_["mean_test_score"]
    data_dict["std_test_score"] = grid_search.cv_results_["std_test_score"]
    data_dict["rank_test_score"] = grid_search.cv_results_["rank_test_score"]

    return data_dict


if __name__ == "__main__":
    #########################
    ## LOAD DATASET
    #########################
    res, labels, metadata = gen_dataset_from_h5("plasticc_balanced_combined_classes_small_ddf")
    bands = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    spatial_comp = int(np.mean([len(ts.observations["flux"]) * 2 for ts in res]))
    time_durations = np.array([ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in res])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    labels_merged = np.array([merged_labels_to_num[merged_labels[x]] for x in labels])
    classes = np.unique(labels_merged)

    ###############################
    ## PRECOMPUTE COUNT VECTORS (configuration fixed)
    #################################
    # configuration name: A
    alph_size = 4
    quantity = "mean"
    num_reduction=True
    irr_handler="supp_interp"
    index_based_paa = False
    mean_bp_dist="normal"
    verbose=False
    _BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    n_jobs = 8
    limit = int(len(res) * 0.05)
    ###################################################
    ###################################################

    wls = [2, 3, 4, 5, 6]
    wins = (mean_time + std_time) * 10 ** np.linspace(-1.95, 0, 40)
    values = []
    out_wins = []
    out_wls = []
    ini = time.time()
    for wl in wls:
        for win in wins:
            threshold = max(1, int(round(wl / 2)))
            tol = wl
            gen = MPTextGenerator(bands=_BANDS, n_jobs=n_jobs, alph_size=4,
                                  quantity=quantity, num_reduction=num_reduction,
                                  irr_handler=irr_handler, index_based_paa=index_based_paa,
                                  mean_bp_dist=mean_bp_dist, verbose=verbose, win=win, word_length=wl)
            vec = CountVectorizer(alph_size=alph_size, word_length=wl, empty_handler=irr_handler, bands=_BANDS)
            corpus = np.array(gen.transform(res))
            fails = 0
            for c in corpus:
                if c is None:
                    fails += 1

            if fails > limit:
                print("%d>%s time series failed to be represented, dropping sequence" % (fails, limit))
                continue
            matrix = sparse.csr_matrix(vec.transform(corpus))
            values.append(matrix)
            out_wins.append(win)
            out_wls.append(wl)
    end = time.time()
    print("PRECOMPUTED TIME: ", end - ini)

    ##########################################
    ## TEST DIFFERENT PIPELINES
    #########################################

    # class_based = True
    # class_type = "type-2"
    # norm_key = "l2"
    # use_idf = True
    smooth_idf = True
    # sublinear_tf = True
    # reducer_type = "lsa"
    n_iter = 20
    n_neighbors = 1

    X = np.arange(len(res))
    data_dict = None

    for class_type in [None, "type-1", "type-2"]:
        for norm_key in ["l2", None]:
            for use_idf in [True, False]:
                for sublinear_tf in [True, False]:
                    for reducer_type in ["lsa", "anova"]:
                        class_based = class_type is not None
                        data_dict_i = pipeline_grid_search(classes, values, out_wins, out_wls,
                                                           spatial_comp, X, labels_merged,
                                                           class_based, class_type, norm_key, use_idf,
                                                           smooth_idf, sublinear_tf, reducer_type)
                        if data_dict is None:
                            data_dict = data_dict_i
                        else:
                            for k, v in data_dict_i.items():
                                data_dict[k] = np.append(data_dict[k], v)

    df = pd.DataFrame(data_dict)
    out_file = os.path.join(main_path, "data", "results", "plasticc", "ibopf_config_A.csv")
    df.to_csv(out_file)



