#!/usr/bin/env python
"""Generate predictions for a dataset using avocado."""
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import argparse
from tqdm import tqdm

import avocado
import random
from ibopf.avocado_adapter import AVOCADOFeaturizer, Dataset, LightGBMClassifier


def process_chunk(classifier, chunk, args, verbose=True):
    # Load the dataset
    if verbose:
        print("Loading dataset...")
    dataset = Dataset.load(args.dataset, metadata_only=True, chunk=chunk,
                           num_chunks=args.num_chunks)
    dataset.set_method("AVOCADO")
    dataset.load_raw_features()
    if args.subset:
        features = dataset.raw_features
        random.seed(chunk)  # directly use the current chunk num as seed, should be ok
        idxs = random.sample(range(features.shape[0]), features.shape[0]//10)
        features = features.iloc[idxs]
        dataset.raw_features = features

    # Generate predictions.
    if verbose:
        print("Generating predictions...")
    predictions = dataset.predict(classifier)

    # Write the predictions to disk.
    if verbose:
        print("Writing out predictions...")
    dataset.write_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        'classifier',
        help='Name of the classifier to use.'
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
        'the data at once. This sets the total number of chunks to use. '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=None,
        help='If set, only process this chunk of the dataset. This is '
        'intended to be used to split processing into multiple jobs.'
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="activate flag if you want to predict on 1/10 of the test set (randomly selected with fixed seed)")

    args = parser.parse_args()

    # Load the classifier
    classifier = LightGBMClassifier.load(args.classifier, method="AVOCADO")

    if args.chunk is not None:
        # Process a single chunk
        process_chunk(classifier, args.chunk, args)
    else:
        # Process all chunks
        print("Processing the dataset in %d chunks..." % args.num_chunks)
        for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                          dynamic_ncols=True):
            process_chunk(classifier, chunk, args, verbose=False)

    print("Done!")
