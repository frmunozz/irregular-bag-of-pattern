import argparse
from ibopf.avocado_adapter import Dataset, MMMBOPFFeaturizer, AVOCADOFeaturizer
from ibopf.similarity_search import KNNSimilaritySearch
import numpy as np
import time
from tqdm import tqdm
from ibopf.settings import get_path, settings
import random
import os
from collections import defaultdict
import pandas as pd
from functools import partial


def main(args):
    k_list = [1, 5, 10, 20, 40]

    # Load the train dataset
    print("Loading train dataset '%s'..." % args.train_dataset)
    dataset = Dataset.load(args.train_dataset, metadata_only=True)

    if args.combine_avocado:
        # load avocado features
        dataset.set_method("AVOCADO")
        dataset.load_raw_features(tag="features_v1")
        avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))


    dataset.set_method(args.method)
    # Load the dataset compact features
    if args.method == "IBOPF":
        if args.use_sparse:
            print("Loading sparse features...")
            dataset.load_sparse_features(features_tag=args.tag)
        else:
            # load compact features
            print("Loading compact features...")
            dataset.load_compact_features(features_tag=args.tag)
    elif args.method == "AVOCADO":
        # load avocado features
        print("Loading raw features...")
        dataset.load_raw_features(tag=args.tag)

    if args.combine_avocado:
        dataset.raw_features = pd.merge(dataset.raw_features, avocado_fea, left_index=True, right_index=True)

    # classes
    object_classes = dataset.metadata["class"] 
    classes = np.unique(object_classes)

    # features
    if args.method == "IBOPF":
        featurizer = MMMBOPFFeaturizer(include_metadata=args.use_metadata)
    elif args.method == "AVOCADO":
        featurizer = AVOCADOFeaturizer(discard_metadata=not args.use_metadata)
    print("INCLUDE METADATA?:", args.use_metadata)

    name = "%s_KNN_SS_%s" % (args.train_dataset, args.tag)
    if args.use_metadata:
        name += "_metadata"

    # similarity search
    ss = KNNSimilaritySearch(name, n_components=np.max(k_list) + 10, metric=args.metric, scale=args.scale)

    print("FIT NEAREST NEIGHBORS FOR %s... " % args.train_dataset, end="")
    features = dataset.select_features(featurizer)
    nan_cols = features.columns[features.isna().any()].to_numpy()
    if len(nan_cols) > 0:
        features = features.drop(columns=nan_cols)
    labels = dataset.metadata["class"]
    ini = time.time()
    ss.fit(features, labels)
    end = time.time()
    print("DONE (time: %.3f secs)" % (end - ini))

    # classifier predict (testset in chunks)
    print("mAP@k FOR %s IN CHUNKS..." % (args.test_dataset))
    test_labels = np.array([])
    aps_computed = defaultdict(list)
    aps_per_class_computed = defaultdict(partial(defaultdict, list))
    for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                      dynamic_ncols=True):
        test_dataset = Dataset.load(args.test_dataset, metadata_only=True, chunk=chunk, num_chunks=args.num_chunks)

        if args.combine_avocado:
            # load avocado features
            test_dataset.set_method("AVOCADO")
            test_dataset.load_raw_features(tag="features_v1")
            avocado_fea = test_dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))


        test_dataset.set_method(args.method)

        # Load the dataset compact features
        if args.method == "IBOPF":
            if args.use_sparse:
                # print("Loading sparse features...")
                test_dataset.load_sparse_features(features_tag=args.tag)
            else:
                # load compact features
                # print("Loading compact features...")
                test_dataset.load_compact_features(features_tag=args.tag)
        elif args.method == "AVOCADO":
            # load avocado features
            # print("Loading raw features...")
            test_dataset.load_raw_features(tag=args.tag)

        if args.combine_avocado:
            test_dataset.raw_features = pd.merge(test_dataset.raw_features, avocado_fea, left_index=True, right_index=True)

        if args.subset:
            features = test_dataset.raw_features
            random.seed(chunk)  # directly use the current chunk num as seed, should be ok
            idxs = random.sample(range(features.shape[0]), features.shape[0]//10)
            features = features.iloc[idxs]
            test_dataset.raw_features = features

        features_test = test_dataset.select_features(featurizer)
        if len(nan_cols):
            features_test = features_test.drop(columns=nan_cols)
        labels_test = test_dataset.metadata["class"]
        aps_dict, aps_per_class_dict = ss.get_map_at_k(features_test, y=labels_test, n_jobs=1, k_list=k_list)
        for k, v in aps_dict.items():
            aps_computed[k].extend(v)
        for k1, sub_dict in aps_per_class_dict.items():
            for k2, v in sub_dict.items():
                aps_per_class_computed[k1][k2].extend(v)
        # break

    f = open(os.path.join(settings["base_path"], "map_results_fixed.csv"), "a+")
    print("%s (metadata:%s) mAP@[1,5,10,20,40]" % (args.method, str(args.use_metadata)), end=" ")
    for k, aps_list in aps_computed.items():
        map_val = np.array(aps_list).mean()
        print(map_val, end=" ")
        f.write("%s,%s,%s,%s,%s,%d,%.2f\n" % (args.method, args.tag, str(args.use_metadata), args.metric, str(args.scale), k, map_val))
    f.close()

    f = open(os.path.join(settings["base_path"], "map_results_fixed_per_class.csv"), "a+")
    for k1, sub_dict in aps_per_class_computed.items():
        for k2, aps_list in sub_dict.items():
            map_val = np.array(aps_list).mean()
            f.write("%s,%s,%s,%s,%s,%d,%d,%.2f\n" % (
            args.method, args.tag, str(args.use_metadata), args.metric, str(args.scale), k2, k1, map_val))
    f.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'train_dataset',
        help='Name of the train dataset to train on.'
    )
    parser.add_argument(
        'test_dataset',
        help='Name of the test dataset to train on.'
    )

    parser.add_argument(
        "-nc",
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )
    parser.add_argument(
        "--tag",
        default="features_v3_LSA",
        type=str,
        help="Use a custom features tag for features h5 file"
    )
    parser.add_argument("--method", default="IBOPF", choices=["IBOPF", "AVOCADO"])
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    # parser.add_argument("--k", type=int, default=10)
    parser.add_argument('--use_metadata', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--use_sparse', action="store_true")
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--combine_avocado", action="store_true")
    args = parser.parse_args()

    main(args)

