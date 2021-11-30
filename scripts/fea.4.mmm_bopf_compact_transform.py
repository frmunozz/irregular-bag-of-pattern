# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import sys
from scipy import sparse
import time

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from src.preprocesing import get_mmbopf_plasticc_path
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
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    main_ini = time.time()
    args = parser.parse_args()

    # get main path (plasticc/MMBOPF)
    main_path = get_mmbopf_plasticc_path()
    # get representation folder (plasticc/MMBOPF/representation)
    repr_directory = os.path.join(main_path, "representation")
    # get representation dataset folder (the dataset should be divided in chunks)
    dataset_repr_directory = os.path.join(repr_directory, "%s_repr_%s" % (args.dataset, args.timestamp))

    # get compact representation folder (plasticc/MMBOPF/compact)
    compact_directory = os.path.join(main_path, "compact")
    # check if exists
    if not os.path.exists(compact_directory):
        os.mkdir(compact_directory)

    # get compact dataset representation folder (the dataset should be divided in chunks)
    dataset_compact_directory = os.path.join(compact_directory, "%s_compact_%s" % (args.dataset, args.timestamp))
    # check if exists
    if not os.path.exists(dataset_compact_directory):
        os.mkdir(dataset_compact_directory)

    print("LOADING COMPACT METHOD MODEL...")
    compact_model_file = os.path.join(compact_directory, "compact_model_%s.pkl" % args.timestamp)
    compact_model = pickle.load(open(compact_model_file, "rb"))

    print("TRANSFORM DATASET (%s_repr_%s) USING COMPACT METHOD BY CHUNKS..." % (args.dataset, args.timestamp))
    # transform test set (all the chunks stored in plasticc/MMBOPF/representation/plasticc_test_repr_timestamp)
    for chunk_id in range(args.num_chunks):
        print("CHUNKS PROCESSED: %d/%d" % (chunk_id, args.num_chunks), end="\r")
        f = os.path.join(dataset_repr_directory, "repr_chunk_%d.npz" % chunk_id)
        chunk_set = sparse.load_npz(f)
        compact = compact_model.transform(chunk_set)
        out_file = os.path.join(dataset_compact_directory, "compact_chunk_%d" % chunk_id)
        np.save(out_file, compact)

    main_end = time.time()
    # save log
    logfile = os.path.join(compact_directory, "log.txt")
    f = open(logfile, "a+")
    f.write("\n**** EXECUTED fea.4.mmm_bopf_compact_transform.py *****\n")
    f.write(".... dataset: %s\n" % args.dataset)
    f.write(".... timestamp: %s\n" % args.timestamp)
    f.write("==== execution time: %.3f sec\n" % (main_end - main_ini))
    f.write("*****************************************\n")
    f.close()

    print("Done!")
