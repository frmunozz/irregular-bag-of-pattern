import argparse
from ibopf.avocado_adapter import Dataset, MMMBOPFFeaturizer, AVOCADOFeaturizer
from ibopf.similarity_search import KNNSimilaritySearch
import numpy as np
import time
from tqdm import tqdm


def main(args):
    # Load the train dataset
    print("Loading train dataset '%s'..." % args.train_dataset)
    dataset = Dataset.load(args.train_dataset, metadata_only=True)
    dataset.set_method(args.method)

    # Load the dataset compact features
    if args.method == "IBOPF":
        # load compact features
        print("Loading compact features...")
        dataset.load_compact_features(features_tag=args.tag)
    elif args.method == "AVOCADO":
        # load avocado features
        print("Loading raw features...")
        dataset.load_raw_features(tag=args.tag)

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
    ss = KNNSimilaritySearch(name, n_components=args.k, metric="euclidean", scale=args.scale)

    print("FIT NEAREST NEIGHBORS FOR %s... " % args.train_dataset, end="")
    features = dataset.select_features(featurizer)
    labels = dataset.metadata["class"]
    ini = time.time()
    ss.fit(features, labels)
    end = time.time()
    print("DONE (time: %.3f secs)" % (end - ini))

    # classifier predict (testset in chunks)
    print("mAP@%d FOR %s IN CHUNKS..." % (args.k, args.test_dataset))
    test_labels = np.array([])
    aps_list = []
    for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                      dynamic_ncols=True):
        test_dataset = Dataset.load(args.test_dataset, metadata_only=True, chunk=chunk, num_chunks=args.num_chunks)
        test_dataset.set_method("IBOPF")
        test_dataset.load_compact_features(features_tag=args.tag)
        features_test = test_dataset.select_features(featurizer)
        labels_test = test_dataset.metadata["class"]
        aps = ss.get_map_at_k(features_test, y=labels_test, n_jobs=1, k=args.k)
        aps_list.extend(aps)
    print("::::::::::::::::::::::::::::::::::::::::::::::")
    print("%s (metadata:%s) mAP@%d : %.2f" % (args.method, str(args.use_metadata), args.k, np.array(aps_list).mean()))
    print("::::::::::::::::::::::::::::::::::::::::::::::")
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
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument('--use_metadata', action='store_true')
    parser.add_argument('--scale', action='store_true')
    args = parser.parse_args()

    main(args)

