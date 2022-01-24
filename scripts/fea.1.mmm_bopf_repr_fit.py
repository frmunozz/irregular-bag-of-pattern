# -*- coding: utf-8 -*-

import argparse
import numpy as np

import avocado

import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from tqdm import tqdm
from src.ibopf.method import IBOPF
from src.timeseries_object import TimeSeriesObject
from sklearn.feature_selection import VarianceThreshold
from scipy import sparse
import time
import pickle
from src.preprocesing import get_mmbopf_plasticc_path

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def process_chunk(chunk, method, args):

    dataset = avocado.load(args.dataset, metadata_only=False,
                           chunk=chunk, num_chunks=args.num_chunks)
    data = []
    labels = []

    for reference_object in tqdm(
            dataset.objects, desc="Reading chunk", dynamic_ncols=True, position=0, leave=True
    ):
        data.append(TimeSeriesObject.from_astronomical_object(reference_object).fast_format_for_numba_code(_BANDS))
        labels.append(reference_object.metadata["class"])
    data = np.array(data)
    labels = np.array(labels)

    chunk_repr = method.mmm_bopf(data, chunk=chunk, position=0, leave=True)
    # chunk_repr = []
    return chunk_repr, labels


if __name__ == '__main__':
    ini = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
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
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
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
        '--chunk',
        type=int,
        default=None,
        help='If set, only process this chunk of the dataset. This is '
             'intended to be used to split processing into multiple jobs.'
    )

    args = parser.parse_args()

    main_path = get_mmbopf_plasticc_path()
    repr_directory = os.path.join(main_path, "representation")
    if not os.path.exists(repr_directory):
        os.mkdir(repr_directory)

    # COMPUTE MMBOPF REPRESENTATION IN RAW FORM (WITHOUT COMPACT METHOD)

    # load method configuration
    method = IBOPF()
    method.config_from_json(os.path.join(main_path, args.config_file))
    method.n_jobs = args.n_jobs
    method.print_ssm_time = True

    # start transforming to MMBOPF repr in chunks
    if args.chunk is not None:
        # Process a single chunk
        data_repr, data_labels = process_chunk(args.chunk, method, args)
    else:
        # Process all chunks
        print("Processing the dataset in %d chunks..." % args.num_chunks)
        chunks_repr = []
        chunks_labels = []
        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            i_repr, i_labels = process_chunk(chunk, method, args)
            chunks_repr.append(i_repr)
            chunks_labels.append(i_labels)

        data_repr = sparse.vstack(chunks_repr, format="csr")
        data_labels = np.hstack(chunks_labels)

    # FIT DROP MODEL ONLY IF COMPACT METHOD IS LSA
    if method.C.lower() == "lsa":
        # fit the variance threshold
        var_threshold = VarianceThreshold()
        var_threshold.fit(data_repr)

        #save variance threshold model
        var_threshold_model_file = os.path.join(repr_directory,
                                                "var_threshold_%s.pkl" % args.timestamp)
        pickle.dump(var_threshold, open(var_threshold_model_file, "wb"))

        # REDUCE REPR BY TRANSFORMING WITH VARIANCE THRESHOLD
        print("shape before variance threshold", data_repr.shape)
        data_repr = var_threshold.transform(data_repr)
        print("shape after variance threshold", data_repr.shape)

    # SAVE THE REPR
    out_repr_file = os.path.join(repr_directory,
                                 "%s_repr_%s" % (args.dataset, args.timestamp))
    sparse.save_npz(out_repr_file, data_repr)

    # SAVE LABELS
    out_lbls_file = os.path.join(repr_directory,
                                 "%s_labels_%s" % (args.dataset, args.timestamp))
    np.save(out_lbls_file, data_labels)
    end = time.time()

    # save log
    logfile = os.path.join(repr_directory, "log.txt")
    f = open(logfile, "a+")
    f.write("\n**** EXECUTED 1.mmm_bopf_repr_train.py *****\n")
    f.write(".... dataset(in): %s\n" % args.dataset)
    f.write(".... config_file: %s\n" % args.config_file)
    f.write(".... timestamp: %s\n" % args.timestamp)
    f.write("==== execution time: %.3f sec\n" % (end - ini))
    f.write("*****************************************\n")
    f.close()

    print("Done!")

    # example execution
    # python save_base_mmm_bopf.py plasticc_augment D:\tesis\data\configs_results\optimal_config\lsa --full_repr_file=agument_full_repr.npz --labels_file=augment_labels.npy