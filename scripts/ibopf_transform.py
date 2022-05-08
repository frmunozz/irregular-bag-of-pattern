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
from ibopf.settings import settings, get_path
from ibopf.avocado_adapter import Dataset, AVOCADOFeaturizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import random

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
    dataset = Dataset.load(args.dataset, metadata_only=False,
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
        write_features(name, df, features_dir="sparse_features_directory",
                       chunk=chunk, num_chunks=args.num_chunks, check_file=False)
        _end = time.time()
        print("Done (time: %.3f secs)" % (_end - _ini))

    _end_chunk = time.time()
    print("[Chunk %d time]: %.3f secs." % (chunk, (_end_chunk - _ini_chunk)))

    return chunk_repr, labels, objs


def load_train_sparse(main_path, name):
    filepath2 = os.path.join(main_path, "sparse_features", name)
    dataset_base = Dataset.load("plasticc_augment_v3", metadata_only=True)
    features_base = avocado.read_dataframe(filepath2, "features")
    sparse_base = sparse.csr_matrix(features_base.values.copy())
    del features_base
    del dataset_base
    return sparse_base


def sparse_features_to_compact(sparse_features, method, args, n_components, sparse_base=None):
    # apply drop zero variance only if is LSA
    # if method.C.upper() != "MANOVA":
    #     print("[Drop zero variance]: applying (train) zero variance model... ")
    #     _ini = time.time()
    #     # we set a new variance threshold pipeline
    #     var_t = ZeroVarianceIBOPF(filename="%s_zero_variance_model.pkl" % args.tag)
    #     var_t.load_pipeline()
    #     sparse_features = var_t.transform(sparse_features)
    #     _end = time.time()
    #     if sparse_base is not None:
    #         sparse_base = var_t.transform(sparse_base)
    #     print("[Drop zero variance]: Done (time: %.3f secs)" % (_end - _ini))

    name_f = method.C.lower()
    min_dist = 0.1
    metric = "euclidean"
    if args.supervised:
        name_f += "_supervised"
    if args.min_dist is not None:
        name_f += "_%f" % args.min_dist
        min_dist = args.min_dist
    if args.metric is not None:
        name_f += "_%s" % args.metric
        metric = args.metric
    if args.densmap:
        name_f += "_densmap"
    if args.combine_avocado:
        name_f += "combined_avocado"

    # apply compact model
    print("[Compact features]: applying (train) compact model '%s'... " % method.C.upper())
    _ini = time.time()
    compact = CompactIBOPF(filename="%s_compact_%s_%s_model.pkl" % (args.tag, name_f, n_components), method=method.C.upper())
    compact.load_pipeline()

    # fit and transform
    ini_compact = time.time()
    # import pdb
    # pdb.set_trace()
    if sparse_base is not None:
        x = compact.pipeline.steps[0][1].transform(sparse_features)
        x = compact.pipeline.steps[1][1].transform(x)
        compact_features = compact.pipeline.steps[2][1].transform(x, basis=sparse_base)
        # compact_features = compact.transform(sparse_features, sparse_base)
    else:
        compact_features = compact.transform(sparse_features, concurrent=False)
    end_compact = time.time()

    if method.C == "MANOVA":
        compact_features = compact_features.toarray()
    _end = time.time()
    print("[Compact features]: Done (time: %.3f secs)" % (_end - _ini))

    return compact_features, end_compact - ini_compact


def compact_models(sparse_features, object_ids, method, chunk, args, sparse_base=None):
    # f = open(os.path.join(settings["IBOPF"]["directory"], "times_compact_test.csv"), "a+")
    f = open(os.path.join(get_path("IBOPF", "directory"), "times_compact_test.csv"), "a+")
    if args.n_components != "max":
        n_components = int(args.n_components)
    else:
        n_components = method.N - 1
    # for C in ["UMAP", "LSA", "PACMAP", "MANOVA"]:
    method.C = args.compact_method.upper()
    compact_features, t_time = sparse_features_to_compact(sparse_features, method, args, n_components, sparse_base=sparse_base)
    name_f = method.C
    min_dist = 0.1
    metric = "euclidean"
    if args.supervised:
        name_f += "_supervised"
    if args.min_dist is not None:
        name_f += "_%f" % args.min_dist
        min_dist = args.min_dist
    if args.metric is not None:
        name_f += "_%s" % args.metric
        metric = args.metric
    if args.densmap:
        name_f += "_densmap"
    if args.combine_avocado:
        name_f += "combined_avocado"

    f.write("%s,%d,%d,%.3f\n" % (name_f, n_components, sparse_features.shape[0], t_time))
    print("[Save %s compact features]: transforming (%d, %d) compact data to AVOCADO format and save... " % (
        method.C, compact_features.shape[0], compact_features.shape[1]))
    _ini = time.time()
    df = pd.DataFrame(compact_features, index=object_ids,
                      columns=["fea" + str(i + 1) for i in range(compact_features.shape[1])])
    df.index.name = "object_id"

    name = "%s_%s_%d_%s.h5" % (args.tag, name_f, n_components, args.dataset)
    write_features(name, df, check_file=False,
                   chunk=chunk, num_chunks=args.num_chunks, features_dir="compact_features_directory")
    _end = time.time()
    print("[Save %s compact features]: Done (time: %.3f secs)" % (method.C, _end - _ini))
    f.close()


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
    parser.add_argument("--compact_method", default="LSA")
    parser.add_argument("--n_components", default="max")
    parser.add_argument("--n_neighbors", default=50, type=int)
    parser.add_argument("--min_dist", default=None, type=float)
    parser.add_argument("--metric", default=None, type=str)
    parser.add_argument("--densmap", action="store_true")
    parser.add_argument("--supervised", action="store_true")

    parser.add_argument("--save_sparse", action='store_true')
    parser.add_argument("--load_sparse", action='store_true')
    parser.add_argument("--sparse_only", action='store_true')
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--combine_avocado", action="store_true")
    args = parser.parse_args()
    main_ini = time.time()

    # load method configuration
    main_path = get_ibopf_plasticc_path()
    # config_file = os.path.join(main_path, args.config_file)
    config_file = get_path("IBOPF", "pipeline_config_file")

    method = IBOPF()
    method.config_from_json(config_file)
    method.print_config()
    method.n_jobs = args.n_jobs

    name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
    filepath = os.path.join(main_path, "sparse_features", name)

    if args.chunk is not None:
        if args.load_sparse:
            dataset = Dataset.load(args.dataset, metadata_only=True, chunk=args.chunk, num_chunks=args.num_chunks)
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

        if args.compact_method.lower() == "pacmap":
            name2 = "%s_%s_plasticc_augment_v3.h5" % (args.tag, method.C)
            sparse_base = load_train_sparse(main_path, name2)
        else:
            sparse_base = None

        for chunk in tqdm(range(args.num_chunks), desc='chunks',
                          dynamic_ncols=True):
            print(">>[chunk %d/%d]" % (chunk + 1, args.num_chunks))

            dataset = Dataset.load(args.dataset, metadata_only=True, chunk=chunk, num_chunks=args.num_chunks)

            if args.combine_avocado:
                dataset.set_method("AVOCADO")
                dataset.load_raw_features(tag="features_v1")
                avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))
                if any(avocado_fea.isna().any()):
                    avocado_fea = avocado_fea.drop(columns=avocado_fea.columns[avocado_fea.isna().any()].tolist())
                dataset.set_method("IBOPF")

            if args.load_sparse:
                features = avocado.read_dataframe(filepath, "features", chunk=chunk, num_chunks=args.num_chunks)
                assert dataset.metadata.values.shape[0] == features.values.shape[0]
                sparse_features = sparse.csr_matrix(features.values)
                object_ids = dataset.metadata.index.to_numpy()
            else:
                sparse_features, _, object_ids = process_chunk_sparse_features(chunk, method, args)

            sparse_split = None
            if args.combine_avocado:
                # import pdb
                # pdb.set_trace()
                # avocado_fea = StandardScaler().fit_transform(avocado_fea.values)
                avocado_fea = sparse.csr_matrix(avocado_fea.values)
                sparse_split = sparse_features.shape[1]
                sparse_features = sparse.hstack([sparse_features, avocado_fea], format="csr")


            if args.subset:
                print("get random subsamples %d -> %d" % (sparse_features.shape[0], sparse_features.shape[0]//10))
                random.seed(chunk)  # directly use the current chunk num as seed, should be ok
                idxs = random.sample(range(sparse_features.shape[0]), sparse_features.shape[0]//10)
                print("reducing data")
                print("shape before", sparse_features.shape, object_ids.shape)
                sparse_features = sparse_features[idxs]
                object_ids = object_ids[idxs]
                print("shape after", sparse_features.shape, object_ids.shape)

            compact_models(sparse_features, object_ids, method, chunk, args, sparse_base=sparse_base)

    main_end = time.time()

    try:
        out_path = settings["IBOPF"]["directory"]
        out_path = get_path("IBOPF", "directory")
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
