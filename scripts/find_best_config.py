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

from src.cross_validation import cv_mmm_bopf
from src.pipeline import MMMBOPFPipeline

'''
This file presents a simple script to run a multi-resolution multi-quantity multi-variate bag-of-patterns features 
algorithm. The selected parameters are:

alphabet_size=4
quantities = [mean, trend, min_max]
IR_scheme= ltc (log TF-IDF normalized)

Optional parameters to be defined by user:

compact method = (LSA/MANOVA)
length of compact vector = (default)500

The script runs a search for the best multi-resolution configuration within the ranges:

word length = [1, 2, 3]
window width = logspace(log(10), log(mean_bandwidth + std_bandwidth), 20)
'''

if __name__ == '__main__':

    set_name = "plasticc_train"

    # read dataset
    dataset, labels_, metadata, split_folds = gen_dataset_from_h5(set_name, num_folds=5)
    split_folds = rearrange_splits(split_folds)
    classes = np.unique(labels_)
    print(len(labels_))
    # estimate spatial complexity
    sc = int(np.mean([len(ts.observations["flux"]) * 2 for ts in dataset]))
    print("the estimated size of each time series is {} [4 bytes units] in average".format(sc))

    # estimate max window of observation
    time_durations = np.array(
        [ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    max_window = mean_time + std_time

    # define some fixed parameters
    alpha = 4
    Q = [["mean", "trend", "min_max"]]
    wls = [1, 2, 3]
    wins = np.logspace(np.log10(10), np.log10(mean_time + std_time), 20)
    print("using window widths: ", wins)

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

    #### USER DEFINED PARAMETER ###########
    C = "MANOVA"  # compact method
    N = sc
    resolution_max = 4
    top_k = 4
    out_path = os.path.join("..", "data", "configs_results", "%s_multi_ress_search" % C.lower())
    if not os.path.exists(out_path):
        raise ValueError("folder doesnt exists")

    # get pipeline
    pipeline = MMMBOPFPipeline(alpha=alpha, Q=Q, C=C, lsa_kw=lsa_kwargs,
                               doc_kw=doc_kwargs, N=N)

    cv_mmm_bopf(dataset, labels_, wins, wls, pipeline, cv=split_folds, resolution_max=resolution_max,
                top_k=top_k, out_path=out_path)


