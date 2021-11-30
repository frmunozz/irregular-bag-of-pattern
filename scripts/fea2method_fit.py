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
    return chunk_repr, labels, objs


# usage example
# python fea2method_fit.py plasticc_augment_v3 optimal_config_lsa -nj 6 -nc 20 -cm LSA -zf zero_var_lsa.pkl -cf compact_lsa.pkl
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
    method.C = args.compact_method

    # start transforming to MMBOPF repr in chunks
    if args.chunk is not None:
        # Process a single chunk
        data_repr, data_labels, object_ids = process_chunk_sparse_features(args.chunk, method, args)
    else:
        # Process all chunks
        print("::CHUNKS>Processing the dataset in %d chunks..." % args.num_chunks)
        chunks_repr = []
        chunks_labels = []
        object_ids = []
        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            i_repr, i_labels, objs = process_chunk_sparse_features(chunk, method, args)
            chunks_repr.append(i_repr)
            chunks_labels.append(i_labels)
            object_ids.append(objs)

        data_repr = sparse.vstack(chunks_repr, format="csr")
        data_labels = np.hstack(chunks_labels)
        object_ids = np.hstack(object_ids)
        print("::CHUNKS>done")

    # apply drop zero variance only if is LSA
    if method.C.upper() == "LSA":
        print("::ZEROVAR>starting to apply zero variance drop")
        # we set a new variance threshold pipeline
        var_t = ZeroVarianceMMMBOPF(filename=args.zerovar_filename)
        var_t.set_pipeline()
        data_repr = var_t.fit_transform(data_repr)
        print(">variance_ mask true count: ", var_t.pipeline.get_support().sum())
        # save the data
        var_t.save_pipeline()
        print("::ZEROVAR>done")

    # apply compact model
    print("::COMPACT>start to apply compact method")
    compact = CompactMMMBOPF(filename=args.compact_filename)

    n_variables = len(_BANDS)
    n_features = data_repr.shape[1]
    classes = np.unique(data_labels)

    # set the pipeline
    compact.set_pipeline(method, n_variables, n_features, classes)

    # fit and transform
    data_repr = compact.fit_transform(data_repr, data_labels)

    if method.C == "MANOVA":
        data_repr = data_repr.toarray()

    # save the pipeline
    compact.save_pipeline()
    print("::COMPACT>done")

    print("RESULTING DATA MATRIX")
    print(">NUMBER OF OBJECTS:", len(data_labels))
    print(">NUMBER OF FEATURES ON SPARSE:", n_features)
    print(">NUMBER OF FEATURES ON COMPACT:", data_repr.shape)

    import pdb
    pdb.set_trace()

    print("::SAVE>transforming data to AVOCADO format and save")
    df = pd.DataFrame(data_repr, index=object_ids, columns=["fea"+str(i+1) for i in range(data_repr.shape[1])])
    df.index.name = "object_id"

    name = "compact_features_%s.h5" % method.C
    write_compact_features(name, df, settings_dir="method_directory")
    print("::SAVE>done")
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
