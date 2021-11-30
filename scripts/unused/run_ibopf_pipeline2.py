import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

from src.preprocesing import gen_dataset_from_h5
from src.pipelines import PipelineBuilder
from src.feature_extraction.text import MPTextGenerator, CountVectorizer


import numpy as np
from scipy import sparse
import time
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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


class MPCVPipeline:
    def __init__(self, classes, spatial_comp, X, labels_merged, class_based, class_type, norm_key, use_idf,
                 smooth_idf, sublinear_tf, reducer_type, n_neighbors=1, n_iter=20):
        self.args = (classes, spatial_comp, X, labels_merged, class_based, class_type, norm_key, use_idf,
                     smooth_idf, sublinear_tf, reducer_type, n_neighbors, n_iter)
        self.c = class_type if class_based else None

        scheme = "l" if sublinear_tf else "n"
        scheme += "t" if use_idf else "n"
        scheme += "c" if norm_key == "l2" else "n"
        scheme += "." + scheme

        self.scheme = scheme

        print("#################################################")
        print("#################################################")
        print("TESTING GRID SEARCH FOR PIPELINE:")
        print("---> CLASS FEATURE: ", self.c)
        print("---> SCHEME: ", scheme)
        print("---> REDUCER TYPE: ", reducer_type)

    def execute(self, arr):
        r = process_map(self.single_cv, arr, max_workers=8, desc="[lineal_pipeline]")

        wins = []
        wls = []
        mean_score = []
        std_score = []
        max_score = []
        for p in r:
            wins.append(p[0])
            wls.append(p[1])
            mean_score.append(p[2])
            std_score.append(p[3])
            max_score.append(p[4])

        data_dict = {
            "class_feature": np.full(len(values), self.c),
            "scheme": np.full(len(values), self.scheme),
            "reducer_type": np.full(len(values), self.args[10]),
            "window": np.array(wins),
            "word_length": np.array(wls),
            "mean_test_score": np.array(mean_score),
            "std_tet_score": np.array(std_score),
            "max_test_score": np.array(max_score)
        }
        return data_dict

    def single_cv(self, pair):
        val, win, wl = pair
        pipeline_builder = PipelineBuilder(class_based=self.args[4], class_type=self.args[5], classes=self.args[0])
        pipeline_builder.set_feature_extraction(precomputed=True, idx=0, data=[val], win_arr=[win], wl_arr=[wl])
        pipeline_builder.set_transformer(norm=self.args[6], use_idf=self.args[7], smooth_idf=self.args[8],
                                         sublinear_tf=self.args[9])
        pipeline_builder.set_reducer(spatial_complexity=self.args[1], reducer_type=self.args[10], n_iter=self.args[12])
        pipeline_builder.set_normalizer()
        pipeline_builder.set_classifier(n_neighbors=self.args[11])
        pipeline = pipeline_builder.build()

        cv_score = cross_val_score(pipeline, self.args[2], y=self.args[3], scoring="balanced_accuracy", cv=10,
                                   verbose=0)

        return [win, wl, np.mean(cv_score), np.std(cv_score), np.max(cv_score)]


def lineal_pipeline(classes, values, out_wins, out_wls, spatial_comp, X, labels_merged,
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

    data_dict = defaultdict(list)
    for idx in tqdm(
                range(len(values)),
                desc="[lineal_pipeline]",
                dynamic_ncols=True):

        pipeline_builder = PipelineBuilder(class_based=class_based, class_type=class_type, classes=classes)
        pipeline_builder.set_feature_extraction(precomputed=True, idx=idx, data=values, win_arr=out_wins, wl_arr=out_wls)
        pipeline_builder.set_transformer(norm=norm_key, use_idf=use_idf, smooth_idf=smooth_idf,
                                         sublinear_tf=sublinear_tf)
        pipeline_builder.set_reducer(spatial_complexity=spatial_comp, reducer_type=reducer_type, n_iter=n_iter)
        pipeline_builder.set_normalizer()
        pipeline_builder.set_classifier(n_neighbors=n_neighbors)
        pipeline = pipeline_builder.build()

        cv_score = cross_val_score(pipeline, X, y=labels_merged, scoring="balanced_accuracy", cv=10, n_jobs=-1, verbose=0)
        data_dict["class_feature"].append(c)
        data_dict["scheme"].append(scheme)
        data_dict["reducer_type"].append(reducer_type)
        data_dict["param_idx"].append(idx)
        data_dict["window"].append(out_wins[idx])
        data_dict["word_length"].append(out_wls[idx])
        data_dict["mean_test_score"].append(np.mean(cv_score))
        data_dict["std_test_score"].append(np.std(cv_score))
        data_dict["max_test_score"].append(np.max(cv_score))

        del pipeline

    return data_dict


def config_pre_computation(config, config_name, res, ini=0, end=-1):
    if len(config_name) == 1:
        wls = [2, 3, 4, 5, 6]
    else:
        wls = [1, 2, 3, 4]

    wins = (mean_time + std_time) * 10 ** np.linspace(-1.95, 0, 40)[ini:end]
    print("#################################################")
    print("#################################################")
    print("TESTING CONFIGURATION ", config_name)

    alph_size = config.get("alph_size")
    quantity = config.get("quantity")
    num_reduction = config.get("num_reduction")
    irr_handler = config.get("irr_handler")
    index_based_paa = config.get("index_based_paa")
    mean_bp_dist = config.get("mean_bp_dist")

    n_jobs = 8
    limit = int(len(res) * 0.05)
    _BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    verbose = False


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
                                  mean_bp_dist=mean_bp_dist, verbose=verbose, win=win, word_length=wl,
                                  tol=tol, threshold=threshold)
            vec = CountVectorizer(alph_size=alph_size, word_length=wl, irr_handler=irr_handler, bands=_BANDS)
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
    return values, out_wins, out_wls


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

    # config_A = {
    #     "alph_size": 4,
    #     "quantity": "mean",
    #     "num_reduction": True,
    #     "irr_handler": "supp_interp",
    #     "index_based_paa": False,
    #     "mean_bp_dist": "normal"
    # }

    config_B = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": True,
        "irr_handler": "#",
        "index_based_paa": False,
        "mean_bp_dist": "normal"
    }

    config_C = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": True,
        "irr_handler": "supp_interp",
        "index_based_paa": True,
        "mean_bp_dist": "normal"
    }

    config_D = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": True,
        "irr_handler": "#",
        "index_based_paa": True,
        "mean_bp_dist": "normal"
    }

    config_E = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": False,
        "irr_handler": "#",
        "index_based_paa": False,
        "mean_bp_dist": "normal"
    }

    config_F = {
        "alph_size": 8,
        "quantity": "mean",
        "num_reduction": True,
        "irr_handler": "supp_interp",
        "index_based_paa": False,
        "mean_bp_dist": "normal"
    }

    config_G = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": True,
        "irr_handler": "supp_interp",
        "index_based_paa": False,
        "mean_bp_dist": "uniform"
    }

    config_H = {
        "alph_size": 4,
        "quantity": "mean",
        "num_reduction": False,
        "irr_handler": "#",
        "index_based_paa": True,
        "mean_bp_dist": "normal"
    }

    config_I = {
        "alph_size": 6,
        "quantity": "mean",
        "num_reduction": False,
        "irr_handler": "#",
        "index_based_paa": True,
        "mean_bp_dist": "normal"
    }

    config_J = {
        "alph_size": 8,
        "quantity": "mean",
        "num_reduction": False,
        "irr_handler": "#",
        "index_based_paa": True,
        "mean_bp_dist": "normal"
    }



    #######################################################################
    ## Test different configurations using best pipeline of prev test
    #######################################################################
    class_based = True
    class_type = "type-2"
    use_idf = True
    sublinear_tf = True
    reducer_type = "lsa"
    norm_key = "l2"
    smooth_idf = True
    n_iter = 20
    n_neighbors = 1

    X = np.arange(len(res))
    out_path = os.path.join(main_path, "data", "results", "plasticc")

    for name_i, config_i in zip(
        ["H", "I", "J"],
        [config_H, config_I, config_J]
    ):
        for ini, end, pi in zip(
                [0],
                [40],
                [0]
        ):
            values, out_wins, out_wls = config_pre_computation(config_i, name_i, res, ini=ini, end=end)
        # pdb.set_trace()
        #     data_dict = pipeline_grid_search(classes, values, out_wins, out_wls,
        #                                  spatial_comp, X, labels_merged,
        #                                  class_based, class_type, norm_key, use_idf,
        #                                  smooth_idf, sublinear_tf, reducer_type)

            data_dict = lineal_pipeline(classes, values, out_wins, out_wls,
                                             spatial_comp, X, labels_merged,
                                             class_based, class_type, norm_key, use_idf,
                                             smooth_idf, sublinear_tf, reducer_type)

            # arr = [[val, win, wl] for val, win, wl in zip(values, out_wins, out_wls)]
            # cv_scorer = MPCVPipeline(classes, spatial_comp, X, labels_merged,
            #                          class_based, class_type, norm_key, use_idf, smooth_idf,
            #                          sublinear_tf, reducer_type)
            # data_dict = cv_scorer.execute(arr)

            df = pd.DataFrame(data_dict)
            df.to_csv(os.path.join(out_path, "ibopf_config_%s_p%s.csv" % (name_i, pi)))




