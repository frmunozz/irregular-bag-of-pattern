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
from src.mmmbopf import write_features, ZeroVarianceMMMBOPF, CompactMMMBOPF, MMMBOPF
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
    _ini_chunk = time.time()
    print(" ")
    print("Loading dataset (Chunk %d)... " % chunk, end="")
    _ini = time.time()
    dataset = avocado.load(args.dataset, metadata_only=False,
                           chunk=chunk, num_chunks=args.num_chunks)
    _end = time.time()
    print("%d AstronomicalObjects loaded (Time: %.3f secs)" % (len(dataset.objects), _end - _ini))
    print("Featurizing the dataset (Chunk %d)... " % chunk)
    objs = dataset.metadata.index.to_numpy()

    chunk_repr, records = method.extended_IBOPF(dataset.objects, record_times=True)

    if zero_variance:
        print("[Drop zero variance]: applying zero variance model... ", end="")
        _ini1 = time.time()
        chunk_repr = zero_variance.transform(chunk_repr)
        _end1 = time.time()
        records["time"] = records["time"] + (_end1 - _ini1) / len(objs)
        print("Done (time: %.3f secs.)" % (_end1 - _ini1))

    print("[Compact features]: applying compact model '%s'... " % method.C.upper(), end="")
    _ini2 = time.time()
    chunk_repr = compact.transform(chunk_repr)
    _end2 = time.time()
    records["time"] = records["time"] + (_end2 - _ini2) / len(objs)
    print("Done (time: %.3f secs.)" % (_end2 - _ini2))
    # print("CHECK", chunk_repr.shape,type(chunk_repr))

    if method.C.upper() == "MANOVA":
        chunk_repr = chunk_repr.toarray()

    print("[Time records]: {} +- {} secs per object".format(np.mean(records["time"]), np.std(records["time"])))

    print("[Save features]: transforming data to AVOCADO format and save... ", end="")
    df = pd.DataFrame(chunk_repr, index=objs, columns=["fea" + str(i + 1) for i in range(chunk_repr.shape[1])])
    df.index.name = "object_id"

    df2 = pd.DataFrame(records, index=objs)
    df2.index.name = "object_id"

    name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
    write_features(name, df, settings_dir="method_directory", chunk=chunk, num_chunks=args.num_chunks,
                   check_file=False, data_records=df2)
    print("Done")
    _end_chunk = time.time()
    print("[Chunk %d time]: %.3f secs." % (chunk, (_end_chunk - _ini_chunk)))


# python fea.method_transform.py plasticc_augment_v3 optimal_config_lsa.json -nj 6 -nc 100 -cm LSA -zf zero_var_lsa.pkl -cf compact_lsa.pkl
# python -m sklearnex fea.method_transform.py plasticc_test optimal_config_lsa.json -nj 6 -nc 200 -cm LSA -zf zero_var_lsa_v3.pkl -cf compact_lsa_v3.pkl --tag=features_v3
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
    parser.add_argument("--tag", type=str, default="features")
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
    method.C = args.compact_method

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
        process_chunk_sparse_features(args.chunk, method, var_t, compact, args)

    else:
        # Process all chunks
        print("[Process dataset]: Processing the dataset in %d chunks..." % args.num_chunks)
        name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            process_chunk_sparse_features(chunk, method, var_t, compact, args)

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