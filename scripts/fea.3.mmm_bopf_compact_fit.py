# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import sys
from scipy import sparse
import time

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from ibopf.pipelines.method import IBOPF
from ibopf.preprocesing import get_ibopf_plasticc_path
import pickle


_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]

if __name__ == '__main__':
    ini = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the train dataset to train on.'
    )
    parser.add_argument(
        "config_file",
        help="filename for method IBOPF configuration"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )
    main_ini = time.time()
    args = parser.parse_args()

    # get main path (plasticc/MMBOPF)
    main_path = get_ibopf_plasticc_path()
    # get representation folder (plasticc/MMBOPF/representation)
    repr_directory = os.path.join(main_path, "representation")
    # get representation test folder (plasticc/MMBOPF/representation/plasticc_test_repr_timestamp)
    test_repr_directory = os.path.join(repr_directory, "%s_repr_%s" % (args.dataset, args.timestamp))

    # get compact representation folder (plasticc/MMBOPF/compact)
    compact_directory = os.path.join(main_path, "compact")
    # check if exists
    if not os.path.exists(compact_directory):
        os.mkdir(compact_directory)

    print("LOADING METHOD CONFIGURATION (%s)..." % args.config_file)
    # load method
    method = IBOPF()
    method.config_from_json(os.path.join(main_path, args.config_file))
    method.print_ssm_time = True

    print("LOADING DATASET (%s_repr_%s)..." % (args.dataset, args.timestamp))
    # load train set
    train_set = sparse.load_npz(os.path.join(repr_directory, "%s_repr_%s.npz" % (args.dataset, args.timestamp)))

    print("LOADING TRAIN SET LABELS (%s_labels_%s)..." % (args.dataset, args.timestamp))
    # load train labels
    train_labels = np.load(os.path.join(repr_directory, "%s_labels_%s.npy" % (args.dataset, args.timestamp)))

    print("GETTING RELEVANT VARIABLES...")
    # get features and variables
    n_variables = len(_BANDS)
    n_features = train_set.shape[1]

    # get classes
    classes = np.unique(train_labels)
    print("==> N_VARIABLES: %d" % n_variables)
    print("==> N_FEATURES: %d" % n_features)
    print("==> TARGET_N_FEATURES: %d" % (method.N - 1))
    print("==> CLASSES: ", classes)

    print("GETTING PIPELINE FOR COMPACT METHOD FIT (%s)..." % method.C)
    # get compact method model
    compact_model = method.get_compact_method_pipeline(n_variables, n_features, classes)

    print("FIT AND TRANSFORM COMPACT METHOD USING DATASET (%s)..." % args.dataset, end="")
    ini = time.time()
    # fit and transform pipeline with train set
    compact = compact_model.fit_transform(train_set, train_labels)
    end = time.time()
    print("DONE (TIME: %.3f secs)" % (end - ini))

    # save compact model
    compact_model_file = os.path.join(compact_directory, "compact_model_%s.pkl" % args.timestamp)
    pickle.dump(compact_model, open(compact_model_file, "wb"))

    # save train compact
    out_file = os.path.join(compact_directory, "%s_compact_%s" % (args.dataset, args.timestamp))
    np.save(out_file, compact)

    main_end = time.time()
    # save log
    logfile = os.path.join(compact_directory, "log.txt")
    f = open(logfile, "a+")
    f.write("\n**** EXECUTED 3.mmm_bopf_compact_fit.py *****\n")
    f.write(".... dataset: %s\n" % args.dataset)
    f.write(".... config_file: %s\n" % args.config_file)
    f.write(".... timestamp: %s\n" % args.timestamp)
    f.write("==== execution time: %.3f sec\n" % (main_end - main_ini))
    f.write("*****************************************\n")
    f.close()

    print("Done!")
