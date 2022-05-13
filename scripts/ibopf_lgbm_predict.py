#!/usr/bin/env python
"""Generate predictions for a dataset using avocado."""
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import argparse
from tqdm import tqdm
import pandas as pd
import avocado
from ibopf.avocado_adapter import MMMBOPFFeaturizer, Dataset, LightGBMClassifier, AVOCADOFeaturizer


def process_chunk(classifier, chunk, args, verbose=True):
    # Load the dataset
    if verbose:
        print("Loading dataset...")
    dataset = Dataset.load(args.dataset, metadata_only=True, chunk=chunk,
                           num_chunks=args.num_chunks
                           # predictions_dir="predictions_mmmbopf_directory"
                           )

    if args.combine_avocado:
        # load avocado features
        dataset.set_method("AVOCADO")
        dataset.load_raw_features(tag="features_v1")
        avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))


    dataset.set_method("IBOPF")
    dataset.load_compact_features(features_tag=args.tag)

    if args.combine_avocado:
        dataset.raw_features = pd.merge(dataset.raw_features, avocado_fea, how="left", left_index=True, right_index=True)

    # drop classes
    classes_to_drop = None
    if args.only_var_stars:
        classes_to_drop = [16, 53, 92]
    elif args.only_supernova:
        classes_to_drop = [42, 52, 62, 64, 67, 90, 95]

    if classes_to_drop is not None:
        dataset.drop_classes(classes_to_drop)

    # Generate predictions.
    if verbose:
        print("Generating predictions...")
    predictions = dataset.predict(classifier)

    # Write the predictions to disk.
    if verbose:
        print("Writing out predictions...")
    dataset.write_predictions(classifier="%s" % (classifier.name))

# python -m sklearnex mmmbopf_predict.py plasticc_test flat_weight --tag=features_v3_MANOVA
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        '--classifier',
        help='Name of the classifier to use.',
        default="lgbm"
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
        "--tag",
        default="compact_LSA_features",
        type=str,
        help="Use a custom features tag for features h5 file"
    )
    parser.add_argument("--combine_avocado", action="store_true")
    parser.add_argument("--only_var_stars", action="store_true")
    parser.add_argument("--only_supernova", action="store_true")

    args = parser.parse_args()

    # Load the classifier
    classifier = LightGBMClassifier.load("%s_%s" % (args.classifier, args.tag))

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
