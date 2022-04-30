# -*- coding: utf-8 -*-

import argparse
import numpy as np

import avocado
import os
import sys
from ibopf.avocado_adapter import IBOPFFeaturizer, AVOCADOFeaturizer, Dataset, LightGBMClassifier
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        'classifier',
        help='Name of the classifier to produce.'
    )
    parser.add_argument(
        '--class_weighting',
        help='Kind of per-class weighting to use. (default: %(default)s)',
        default='flat',
        choices=('flat', 'kaggle'),
    )
    parser.add_argument(
        '--object_weighting',
        help='Kind of per-object weighting to use. (default: %(default)s)',
        default='flat',
        choices=('flat', 'redshift'),
    )
    parser.add_argument(
        '--simulate_plasticc_bias',
        help='Simulate adding a bias for the PLAsTiCC dataset.',
        default=None,
        choices=(None, 'low_redshift', 'high_redshift'),
    )
    parser.add_argument(
        "--tag",
        default="compact_LSA_features",
        type=str,
        help="Use a custom features tag for features h5 file"
    )

    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset '%s'..." % args.dataset)
    dataset = Dataset.load(args.dataset, metadata_only=True, predictions_dir="predictions_mmmbopf_directory")

    # Load the dataset raw features
    print("Loading raw features...")
    raw_features_ibopf = dataset.load_compact_features(features_tag=args.tag)
    selected_features_ibopf = dataset.select_features(MMMBOPFFeaturizer(include_metadata=False))

    # Load the dataset raw features
    print("Loading raw features...")
    raw_features_avocado = dataset.load_raw_features()
    selected_features_avocado = dataset.select_features(avocado.plasticc.PlasticcFeaturizer())

    dataset.features = pd.concat([selected_features_ibopf, selected_features_avocado], axis=1)

    # Figure out which weightings to use.
    if args.class_weighting == 'flat':
        class_weights = None
    elif args.class_weighting == 'kaggle':
        class_weights = avocado.plasticc.plasticc_kaggle_weights
    else:
        raise avocado.AvocadoException("Invalid class weighting '%s'!" %
                                       args.weighting)

    if args.object_weighting == 'flat':
        weighting_function = avocado.evaluate_weights_flat
    elif args.object_weighting == 'redshift':
        weighting_function = avocado.evaluate_weights_redshift
    else:
        raise avocado.AvocadoException("Invalid object weighting '%s'!" %
                                       args.weighting)

    # Train the classifier
    print("Training classifier '%s'..." % args.classifier)
    classifier = LightGBMClassifier(
        "%s_%s" % (args.classifier, args.tag),
        None,
        class_weights=class_weights,
        weighting_function=weighting_function,
        settings_dir="predictions_mmmbopf_directory",
        use_existing_features=True
    )
    print("out path:", classifier.path, classifier)
    classifier.train(dataset)

    # Save the classifier
    print("Saving the classifier...")
    classifier.write(overwrite=True)

    print("Done!")