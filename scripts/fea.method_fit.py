# -*- coding: utf-8 -*-

import argparse
import numpy as np

import avocado

import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from tqdm import tqdm
from ibopf.timeseries_object import TimeSeriesObject
from scipy import sparse
import time
import pickle
from ibopf.preprocesing import get_ibopf_plasticc_path
from ibopf.pipelines import write_features, ZeroVarianceIBOPF, CompactIBOPF, IBOPF

from sklearn.feature_selection import VarianceThreshold
import pandas as pd


"""
    transform the train-set to compact representation form
    The script do:
    - load the train-set in chunks and extract sparse features
    - apply drop_zero_variance if is LSA and save the model
    - apply compactMMMBOPF in order to generate the compact features, and save the model
    - save the compact features in .h5 format file using AVOCADO library

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
    _end_chunk = time.time()
    print("[Chunk %d time]: %.3f secs." % (chunk, (_end_chunk - _ini_chunk)))

    return chunk_repr, labels, objs


def sparse_features_to_compact(sparse_features, labels, method):

    # apply drop zero variance only if is LSA
    if method.C.upper() == "LSA":
        print("[Drop zero variance]: applying (train) zero variance model... ", end="")
        _ini = time.time()
        # we set a new variance threshold pipeline
        var_t = ZeroVarianceIBOPF(filename=args.zerovar_filename)
        var_t.set_pipeline()
        sparse_features = var_t.fit_transform(sparse_features)
        # save the data
        var_t.save_pipeline()
        _end = time.time()
        print("Done (time: %.3f secs)" % (_end - _ini))

    # apply compact model
    print("[Compact features]: applying (train) compact model '%s'... " % method.C.upper(), end="")
    _ini = time.time()
    compact = CompactIBOPF(filename=args.compact_filename, method=method.C.upper())

    n_variables = len(_BANDS)
    n_features = sparse_features.shape[1]
    classes = np.unique(data_labels)

    # set the pipeline
    compact.set_pipeline(method, n_variables, n_features, classes)

    # fit and transform
    compact_features = compact.fit_transform(sparse_features, data_labels)

    if method.C == "MANOVA":
        compact_features = compact_features.toarray()

    # save the pipeline
    compact.save_pipeline()
    _end = time.time()
    print("Done (time: %.3f secs)" % (_end - _ini))

    return compact_features


# usage example
# python fea.method_fit.py plasticc_augment_v3 optimal_config_lsa -nj 6 -nc 20 -cm LSA -zf zero_var_lsa.pkl -cf compact_lsa.pkl
# python -m sklearnex fea.method_fit.py plasticc_augment_v3 optimal_config_lsa.json -nj 6 -nc 5 -cm LSA -zf zero_var_lsa_v3.pkl -cf compact_lsa_v3.pkl --tag=features_v3
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
    main_path = get_ibopf_plasticc_path()
    config_file = os.path.join(main_path, args.config_file)

    method = IBOPF()
    method.config_from_json(config_file)
    method.n_jobs = args.n_jobs
    method.print_ssm_time = True
    method.C = args.compact_method

    # start transforming to MMBOPF repr in chunks
    if args.chunk is not None:
        # Process a single chunk
        sparse_features, data_labels, object_ids = process_chunk_sparse_features(args.chunk, method, args)
    else:
        # Process all chunks
        print("[Process dataset]: Processing the dataset in %d chunks..." % args.num_chunks)
        chunks_features = []
        chunks_labels = []
        object_ids = []
        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            i_repr, i_labels, objs = process_chunk_sparse_features(chunk, method, args)
            chunks_features.append(i_repr)
            chunks_labels.append(i_labels)
            object_ids.append(objs)

        sparse_features = sparse.vstack(chunks_features, format="csr")
        data_labels = np.hstack(chunks_labels)
        object_ids = np.hstack(object_ids)

    compact_features = sparse_features_to_compact(sparse_features, data_labels, method)

    print("[Save features]: transforming data to AVOCADO format and save... ", end="")
    _ini = time.time()
    df = pd.DataFrame(compact_features, index=object_ids, columns=["fea"+str(i+1) for i in range(compact_features.shape[1])])
    df.index.name = "object_id"

    name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
    write_features(name, df, settings_dir="method_directory")
    _end = time.time()
    print("Done (time: %.3f secs)" % (_end - _ini))
    main_end = time.time()

    try:
        out_path = avocado.settings["method_directory"]
        f = open(os.path.join(out_path, "log.txt"), "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("script_name: fea.method_fit.py\n")
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
