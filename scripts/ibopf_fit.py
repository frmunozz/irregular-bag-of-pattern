# -*- coding: utf-8 -*-

import argparse
import numpy as np
import avocado
import os
from tqdm import tqdm
from scipy import sparse
import time
from ibopf.preprocesing import get_ibopf_plasticc_path
from ibopf.pipelines import write_features, ZeroVarianceIBOPF, CompactIBOPF, IBOPF
from ibopf.avocado_adapter import Dataset, AVOCADOFeaturizer
from ibopf.settings import settings, get_path
from sklearn.preprocessing import StandardScaler

import pandas as pd


"""
    fit the models to a train set and transform it to compact feature vector
    the script do:
    - load the dataset in chunks and extract sparse features
    - (optional) save the sparse features using .h5 file (can be huge)
    - (optional) directly load the sparse features
    - fit the compact method and save the models to disk
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
        print("[Save sparse features]: transforming %s sparse data to AVOCADO format and save... " % chunk_repr.shape[1], end="")
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


def sparse_features_to_compact(sparse_features, data_labels, method, args, n_neighbors=15, sparse_split=None):

    # apply drop zero variance only if is not MANOVA
    # if method.C.upper() != "MANOVA":
    #     print("[Drop zero variance]: applying (train) zero variance model... ")
    #     _ini = time.time()
    #     # we set a new variance threshold pipeline
    #     var_t = ZeroVarianceIBOPF(filename="%s_zero_variance_model.pkl" % args.tag)
    #     var_t.set_pipeline()
    #     sparse_features = var_t.fit_transform(sparse_features)
    #     # save the data
    #     var_t.save_pipeline(overwrite=True)
    #     _end = time.time()
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
    compact = CompactIBOPF(filename="%s_compact_%s_%d_model.pkl" % (args.tag, name_f, method.N-1), method=method.C.upper())

    n_variables = len(_BANDS)
    n_features = sparse_features.shape[1]
    classes = np.unique(data_labels)
    if method.C == "MANOVA":
        method.N = method.N * n_variables

    # set the pipeline
    compact.set_pipeline(method, n_variables, n_features, classes, n_neighbors=n_neighbors,
                         min_dist=min_dist, metric=metric, densmap=args.densmap, sparse_split=sparse_split)

    # fit and transform
    ini_compact = time.time()
    if method.C.lower() == "manova" or args.supervised:
        compact_features = compact.fit_transform(sparse_features, y=data_labels)
    else:
        compact_features = compact.fit_transform(sparse_features)
    end_compact = time.time()

    if method.C == "MANOVA":
        compact_features = compact_features.toarray()

    # save the pipeline
    compact.save_pipeline(overwrite=True)
    _end = time.time()
    print("[Compact features]: Done (time: %.3f secs)" % (_end - _ini))

    return compact_features, ini_compact, end_compact


# usage example
# python -m sklearnex fea.ibopf_fit.py plasticc_augment_v3 optimal_config_lsa.json -nc 100 -nj 6 --tag=features_v3 --save_sparse=True
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
    parser.add_argument("--compact_method", default="lsa", type=str)
    parser.add_argument("--n_components", default="max", type=str)
    parser.add_argument("--n_neighbors", default=50, type=int)
    parser.add_argument("--min_dist", default=None, type=float)
    parser.add_argument("--metric", default=None, type=str)
    parser.add_argument("--densmap", action="store_true")
    parser.add_argument("--supervised", action="store_true")

    parser.add_argument("--save_sparse", action='store_true')
    parser.add_argument("--load_sparse", action='store_true')
    parser.add_argument("--sparse_only", action='store_true')
    parser.add_argument("--combine_avocado", action="store_true")
    args = parser.parse_args()
    print("start script")
    main_ini = time.time()

    # load method configuration
    main_path = get_ibopf_plasticc_path()
    # config_file = os.path.join(main_path, args.config_file)
    config_file = get_path("IBOPF", "pipeline_config_file")

    method = IBOPF()
    method.config_from_json(config_file)
    method.n_jobs = args.n_jobs

    # method.print_config()
    print("Loading dataset metadata full... ", end="")
    _ini = time.time()
    dataset = Dataset.load(args.dataset, metadata_only=True)
    _end = time.time()
    print("%d Objects metadata loaded (Time: %.3f secs)" % (dataset.metadata.values.shape[0], _end - _ini))

    if args.combine_avocado:
        dataset.set_method("AVOCADO")
        dataset.load_raw_features(tag="features_v1")
        avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))
        avocado_fea = avocado_fea.dropna(axis=1)
        # if any(avocado_fea.isna().any()):
            # avocado_fea = avocado_fea.drop(columns=avocado_fea.columns[avocado_fea.isna().any()].tolist())


    # load sparse repr or generate it depending on flag
    if args.load_sparse:
        # load the sparse data stored in /sparse_features
        # in this case the whole data is loaded at once
        # BE CAREFUL!
        name = "%s_%s_%s.h5" % (args.tag, method.C, args.dataset)
        filepath = os.path.join(main_path, "sparse_features", name)

        print("Loading sparse features full...", end="")
        _ini = time.time()
        features = avocado.read_dataframe(filepath, "features")
        _end = time.time()
        print("%d Objects sparse feature loaded (Time: %.3f secs)" % (features.values.shape[0], _end - _ini))

        sparse_features = sparse.csr_matrix(features.values)
        data_labels = dataset.metadata["class"].to_numpy()
        object_ids = dataset.metadata.index.to_numpy()

    else:
        # start transforming to sparse repr in chunks
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

    if not args.sparse_only:
        # out_path = settings["IBOPF"]["directory"]
        out_path = get_path("IBOPF", "directory")
        f_compacts_exists = os.path.exists(os.path.join(out_path, "compact_times.csv"))
        f_compacts = open(os.path.join(out_path, "compact_times.csv"), "a+")
        if not f_compacts_exists:
            f_compacts.write("compact_method,n_components,n_neighbors,fit_trasform_time\n")
        # for C in ["LSA", "MANOVA", "UMAP", "PACMAP"]:
        # sparse_features_arr = sparse_features.toarray()

        if args.n_components != "max":
            method.N = int(args.n_components) + 1

        method.C = args.compact_method

        sparse_split = None
        if args.combine_avocado:
            # import pdb
            # pdb.set_trace()
            # avocado_fea = StandardScaler().fit_transform(avocado_fea.values)
            avocado_fea = sparse.csr_matrix(avocado_fea.values)
            sparse_split = sparse_features.shape[1]
            sparse_features = sparse.hstack([sparse_features, avocado_fea], format="csr")


        compact_features, ini_compact, end_compact = sparse_features_to_compact(
            sparse_features, data_labels, method, args, n_neighbors=args.n_neighbors, sparse_split=sparse_split)
        
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

        f_compacts.write("%s,%d,%d,%.3f\n" % (name_f, method.N-1, args.n_neighbors, end_compact - ini_compact))
        print("[Save %s compact features]: transforming %s compact data to AVOCADO format and save... " % (
        method.C, compact_features.shape[1]))
        _ini = time.time()
        df = pd.DataFrame(compact_features, index=object_ids,
                          columns=["fea" + str(i + 1) for i in range(compact_features.shape[1])])
        df.index.name = "object_id"

        name = "%s_%s_%d_%s.h5" % (args.tag, name_f, method.N-1, args.dataset)

        write_features(name, df, method="IBOPF", overwrite=True, features_dir="compact_features_directory")
        _end = time.time()
        print("[Save %s compact features]: Done (time: %.3f secs)" % (method.C, _end - _ini))
        f_compacts.close()

    main_end = time.time()

    try:
        # out_path = settings["IBOPF"]["directory"]
        out_path = get_path("IBOPF", "directory")
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
