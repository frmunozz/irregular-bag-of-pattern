# -*- coding: utf-8 -*-
import argparse
import numpy as np

import avocado

import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from tqdm import tqdm
from src.timeseries_object import TimeSeriesObject
from scipy import sparse
import time
import pickle
from src.preprocesing import get_mmbopf_plasticc_path
from src.mmmbopf import write_compact_features, ZeroVarianceMMMBOPF, CompactMMMBOPF, MMMBOPF
from src.avocado_adapter import *

from sklearn.feature_selection import VarianceThreshold
import pandas as pd


"""
    transform the test-set to compact representation form
    
     - load the zerovariance model and compact model 
     - process a chunk of the set and do:
        - compute the sparse representation
        - apply zero-variance model
        - apply compact model
        - save the compact features in .h5 format file using AVOCADO library

    It uses the default bands for PLaSTiCC.
"""


_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


# process the train set in chunks
def process_chunk_sparse_features(chunk, method, zero_variance, compact, args):

    dataset = avocado.load(args.dataset, metadata_only=False,
                           chunk=chunk, num_chunks=args.num_chunks)
    data = []
    labels = []
    objs = []

    for reference_object in tqdm(
            dataset.objects, desc="Reading chunk", dynamic_ncols=True, position=0, leave=True
    ):
        data.append(TimeSeriesObject.from_astronomical_object(reference_object).fast_format_for_numba_code(_BANDS))
        labels.append(reference_object.metadata["class"])
        objs.append(reference_object.metadata["object_id"])
    data = np.array(data)
    labels = np.array(labels)
    objs = np.array(objs)

    chunk_repr = method.mmm_bopf(data, chunk=chunk)
    if zero_variance:
        print("::CHUNK=%d>::ZEROVAR>applying zero variance model...", end="")
        _ini1 = time.time()
        chunk_repr = zero_variance.transform(chunk_repr)
        _end1 = time.time()
        print("Done (time: %.3f)" % (_end1 - _ini1))

    print("::CHUNK=%d>::COMPACT>applying compact model...", end="")
    _ini2 = time.time()
    chunk_repr = compact.transform(chunk_repr)
    _end2 = time.time()
    print("Done (time: %.3f)" % (_end2 - _ini2))

    return chunk_repr, labels, objs


# python fea.method_transform.py plasticc_augment_v3 optimal_config_lsa.json -nj 6 -nc 100 -cm LSA -zf zero_var_lsa.pkl -cf compact_lsa.pkl
if __name__ == '__main__':
    ini = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        "config_file",
        help="filename for method MMMBOPF configuration"
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
    )
    parser.add_argument(
        '-nc',
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )
    parser.add_argument(
        '-ch',
        '--chunk',
        type=int,
        default=None,
        help='If set, only process this chunk of the dataset. This is '
             'intended to be used to split processing into multiple jobs.'
    )
    parser.add_argument('-zf', '--zerovar_filename', type=str, default="zero_variance_model.pkl")
    parser.add_argument('-cf', '--compact_filename', type=str, default="compact_pipeline.pkl")
    parser.add_argument('-cm', '--compact_method', type=str, default="LSA")
    args = parser.parse_args()
    main_ini = time.time()

    # load method configuration
    main_path = get_mmbopf_plasticc_path()
    config_file = os.path.join(main_path, args.config_file)

    method = MMMBOPF()
    method.config_from_json(config_file)
    method.n_jobs = args.n_jobs
    method.print_ssm_time = True

    if method.C.upper() == "LSA":
        print("LOADING ZERO VARIANCE")
        var_t = ZeroVarianceMMMBOPF(filename=args.zerovar_filename)
        var_t.load_pipeline()
        print("DONE")
    else:
        var_t = None

    compact = CompactMMMBOPF(filename=args.compact_filename)
    compact.load_pipeline()

    # start transforming to MMBOPF repr in chunks
    if args.chunk is not None:
        # Process a single chunk
        data_repr, data_labels, object_ids = process_chunk_sparse_features(args.chunk, method, var_t, compact, args)
    else:
        # Process all chunks
        print("::CHUNKS>Processing the dataset in %d chunks..." % args.num_chunks)
        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            _ini_chunk = time.time()
            i_repr, i_labels, objs = process_chunk_sparse_features(chunk, method, var_t, compact, args)

            print("::CHUNK=%d>RESULTING DATA MATRIX" % chunk)
            print("::CHUNK=%d>NUMBER OF OBJECTS:" % chunk, len(i_labels))
            print("::CHUNK=%d>NUMBER OF FEATURES ON COMPACT:" % chunk, i_repr.shape)

            print("::CHUNK=%d>::SAVE>transforming data to AVOCADO format and save" % chunk)
            df = pd.DataFrame(i_repr, index=objs, columns=["fea" + str(i + 1) for i in range(i_repr.shape[1])])
            df.index.name = "object_id"

            name = "features_%s_%s.h5" % (method.C, args.dataset)
            write_compact_features(name, df, settings_dir="method_directory", chunk=chunk, num_chunks=args.num_chunks,
                                   check_file=False)
            print("::CHUNK=%d>::SAVE>done")
            _end_chunk = time.time()
            print("::CHUNK=%d>time: %.3f" % (chunk, (_end_chunk - _ini_chunk)))

        print("::CHUNKS>done")

    main_end = time.time()
    try:
        out_path = avocado.settings["method_directory"]
        f = open(os.path.join(out_path, "log.txt"), "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("script_name: fea.method_transform.py\n")
        f.write("compact_method: %s\n" % method.C),
        f.write("alpha: %d\n" % method.alpha),
        f.write("quantities: %s\n" % method.quantities_code()),
        f.write("resolutions: %s\n" % str(method.R))
        f.write("execution time: %.3f\n" % (main_end - main_ini))
        f.write("++++++++++++++++++++++++++++++++\n")
        f.close()
    except Exception as e:
        print("failed to write log file, error: ", e)
        try:
            f.close()
        except:
            pass