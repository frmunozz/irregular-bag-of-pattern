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
from src.ibopf import write_features, ZeroVarianceIBOPF, CompactIBOPF, IBOPF

from sklearn.feature_selection import VarianceThreshold
import pandas as pd

"""
    transform a dataset to compact feature vector
    the script do:
    - load the dataset in chunks and extract sparse features
    - (optional) save the sparse features using .h5 file (can be huge)
    - (optional) load the sparse features in chunks
    - load the models for the compact vectors from disk
    - transform the dataset to compact features and save in .h5 format.
    
    It uses the default bands for PLaSTiCC.
"""

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


# process the train set in chunks
def process_chunk_sparse_features(chunk, method, args):
    _ini_chunk = time.time()
    print(" ")
    print("Loading dataset (Chunk %d)... " % chunk, end="")
    _ini = time.time()
    dataset = avocado.load(args.dataset, metadata_only=False,
                           chunk=chunk, num_chunks=args.num_chunks)
    _end = time.time()
    print("%d AstronomicalObjects loaded (Time: %.3f secs)" % (len(dataset.objects), _end - _ini))
    print("Featurizing the dataset (Chunk %d)... " % chunk)
    labels = dataset.metadata["class"].to_numpy()
    objs = dataset.metadata.index.to_numpy()
    chunk_repr = method.extended_IBOPF(dataset.objects, record_times=False)
    if args.save_sparse:
        print(
            "[Save sparse features]: transforming %s sparse data to AVOCADO format and save... " % chunk_repr.shape[1],
            end="")
        _ini = time.time()
        df = pd.DataFrame(chunk_repr.toarray(), index=objs,
                          columns=["fea" + str(i + 1) for i in range(chunk_repr.shape[1])])
        df.index.name = "object_id"

        name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
        write_features(name, df, settings_dir="method_directory", features_dir="sparse_features",
                       chunk=chunk, num_chunks=args.num_chunks, check_file=False)
        _end = time.time()
        print("Done (time: %.3f secs)" % (_end - _ini))

    _end_chunk = time.time()
    print("[Chunk %d time]: %.3f secs." % (chunk, (_end_chunk - _ini_chunk)))

    return chunk_repr, labels, objs


def sparse_features_to_compact(sparse_features, method, args):
    # apply drop zero variance only if is LSA
    if method.C.upper() == "LSA":
        print("[Drop zero variance]: applying (train) zero variance model... ")
        _ini = time.time()
        # we set a new variance threshold pipeline
        var_t = ZeroVarianceIBOPF(filename="%s_zero_variance_model.pkl" % args.tag)
        var_t.load_pipeline()
        sparse_features = var_t.transform(sparse_features)
        _end = time.time()
        print("[Drop zero variance]: Done (time: %.3f secs)" % (_end - _ini))

    # apply compact model
    print("[Compact features]: applying (train) compact model '%s'... " % method.C.upper())
    _ini = time.time()
    compact = CompactIBOPF(filename="%s_compact_%s_model.pkl" % (args.tag, method.C.lower()), method=method.C.upper())
    compact.load_pipeline()

    # fit and transform
    compact_features = compact.transform(sparse_features)

    if method.C == "MANOVA":
        compact_features = compact_features.toarray()
    _end = time.time()
    print("[Compact features]: Done (time: %.3f secs)" % (_end - _ini))

    return compact_features


def compact_models(sparse_features, object_ids, method, chunk, args):
    for C in ["LSA", "MANOVA"]:
        method.C = C
        compact_features = sparse_features_to_compact(sparse_features, method, args)

        print("[Save %s compact features]: transforming %s compact data to AVOCADO format and save... " % (
            C, compact_features.shape[1]))
        _ini = time.time()
        df = pd.DataFrame(compact_features, index=object_ids,
                          columns=["fea" + str(i + 1) for i in range(compact_features.shape[1])])
        df.index.name = "object_id"

        name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
        write_features(name, df, settings_dir="method_directory", check_file=False,
                       chunk=chunk, num_chunks=args.num_chunks)
        _end = time.time()
        print("[Save %s compact features]: Done (time: %.3f secs)" % (C, _end - _ini))


# usage example
# python -m sklearnex fea.ibopf_transform.py plasticc_test optimal_config_lsa.json -nc 100 -nj 6 --tag=features_v3 --save_sparse=True
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

    parser.add_argument(
        "--tag",
        default="features",
        type=str,
        help="Use a custom features tag for features h5 file"
    )

    parser.add_argument("--save_sparse", action='store_true')
    parser.add_argument("--load_sparse", action='store_true')
    parser.add_argument("--sparse_only", action='store_true')
    args = parser.parse_args()
    main_ini = time.time()

    # load method configuration
    main_path = get_mmbopf_plasticc_path()
    config_file = os.path.join(main_path, args.config_file)

    method = IBOPF()
    method.config_from_json(config_file)
    method.print_config()
    method.n_jobs = args.n_jobs

    name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
    filepath = os.path.join(main_path, "sparse_features", name)

    if args.chunk is not None:
        if args.load_sparse:
            dataset = avocado.load(args.dataset, metadata_only=True, chunk=args.chunk, num_chunks=args.num_chunks)
            features = avocado.read_dataframe(filepath, "features", chunk=args.chunk, num_chunks=args.num_chunks)
            assert dataset.metadata.values.shape[0] == features.values.shape[0]
            sparse_features = sparse.csr_matrix(features.values)
            object_ids = dataset.metadata.index.to_numpy()
        else:
            sparse_features, _, object_ids = process_chunk_sparse_features(args.chunk, method, args)
        compact_models(sparse_features, object_ids, method, args.chunk, args)
    else:
        if args.load_sparse:
            print("[load sparse data]: Processing the dataset in %d chunks..." % args.num_chunks)
        else:
            print("[Process dataset]: Processing the dataset in %d chunks..." % args.num_chunks)

        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            print(">>[chunk %d/%d]" % (chunk + 1, args.num_chunks))
            if args.load_sparse:
                dataset = avocado.load(args.dataset, metadata_only=True, chunk=chunk, num_chunks=args.num_chunks)
                features = avocado.read_dataframe(filepath, "features", chunk=chunk, num_chunks=args.num_chunks)
                assert dataset.metadata.values.shape[0] == features.values.shape[0]
                sparse_features = sparse.csr_matrix(features.values)
                object_ids = dataset.metadata.index.to_numpy()
            else:
                sparse_features, _, object_ids = process_chunk_sparse_features(chunk, method, args)

            compact_models(sparse_features, object_ids, method, chunk, args)

    main_end = time.time()

    try:
        out_path = avocado.settings["method_directory"]
        f = open(os.path.join(out_path, "log.txt"), "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("script_name: fea.ibopf_transform.py\n")
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
