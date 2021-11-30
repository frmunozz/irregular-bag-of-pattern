# -*- coding: utf-8 -*-
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

# from src.classifiers import prototype_knn
from src.avocado_adapter import MMMBOPFFeaturizer, KNNClassifier, Dataset
import avocado
import time

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
        "-m",
        '--use_metadata',
        default=None,
        type=str,
        help="Use the optional metadata on the classifier"
    )

    parser.add_argument(
        "-p",
        '--prototype',
        default=None,
        type=str,
        help="Use prototypes to compute the KNN classifier"
    )

    parser.add_argument(
        "-n",
        '--normalizer',
        default=None,
        type=str,
        help="Use Normalizer on the pipeline"
    )

    parser.add_argument(
        "-s",
        '--scaler',
        default=None,
        type=str,
        help="Use StandarScaler on the pipeline"
    )
    parser.add_argument(
        "--tag",
        default="compact_LSA_features",
        type=str,
        help="Use a custom features tag for features h5 file"
    )
    # python mmmbopf_knn_direct.py plasticc_augment_v3 plasticc_test --prototype=True --tag=features_LSA
    args = parser.parse_args()

    # Load the train dataset
    print("Loading train dataset '%s'..." % args.train_dataset)
    dataset = Dataset.load(args.train_dataset, metadata_only=True, predictions_dir="predictions_mmmbopf_directory")

    # Load the dataset compact features
    print("Loading raw features...")
    dataset.load_compact_features(features_tag=args.tag)

    # classes
    object_classes = dataset.metadata["class"]
    classes = np.unique(object_classes)

    # features
    featurizer = MMMBOPFFeaturizer(include_metadata=args.use_metadata is not None)
    print("INCLUDE METADATA?:", featurizer.include_metadata)

    # classifier train
    name = args.train_dataset + "_K-NN"
    if args.use_metadata is not None:
        name += "_metadata"

    classifier = KNNClassifier(name, featurizer, prototype=args.prototype is not None,
                               normalizer=args.normalizer is not None,
                               scaler=args.scaler is not None)
    print("PROTOTYPE:", classifier.prototype)
    print("NORMALIZER:", classifier.normalizer)
    print("SCALER:", classifier.scaler)

    print("TRAIN CLASSIFIER FOR %s... " % args.train_dataset, end="")
    ini = time.time()
    classifier.train(dataset)
    end = time.time()
    print("DONE (time: %.3f secs)" % (end - ini))

    # classifier predict (testset in chunks)
    print("PREDICT CLASSES FOR %s IN CHUNKS..." % args.test_dataset)
    prototype_lbl = "prototype" if args.prototype is not None else "not_prototype"
    test_labels = np.array([])
    for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                      dynamic_ncols=True):
        test_dataset = Dataset.load(args.test_dataset, metadata_only=True, chunk=chunk,
                                    num_chunks=args.num_chunks, predictions_dir="predictions_mmmbopf_directory")
        test_dataset.load_compact_features(features_tag=args.tag)

        predictions = test_dataset.predict(classifier)
        test_dataset.write_predictions(classifier=classifier.name)

    print("Done!")