# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Train a classifier using avocado.

For now, this only supports a LightGBM classifier with the PLAsTiCC featurizer.
"""

import argparse
import numpy as np
import pandas as pd
import avocado
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from ibopf.avocado_adapter import MMMBOPFFeaturizer, Dataset, LightGBMClassifier, AVOCADOFeaturizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        '--classifier',
        help='Name of the classifier to produce.',
        default="lgbm"
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
    parser.add_argument('--use_metadata', action='store_true')
    parser.add_argument("--combine_avocado", action="store_true")
    parser.add_argument("--only_var_stars", action="store_true")
    parser.add_argument("--only_supernova", action="store_true")

    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset '%s'..." % args.dataset)
    dataset = Dataset.load(args.dataset, metadata_only=True)

    if args.combine_avocado:
        # load avocado features
        dataset.set_method("AVOCADO")
        dataset.load_raw_features(tag="features_v1")
        avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=True))

    dataset.set_method("IBOPF")
    # Load the dataset raw features
    print("Loading raw features...")
    dataset.load_compact_features(features_tag=args.tag)

    if args.combine_avocado:
        dataset.raw_features = pd.merge(dataset.raw_features, avocado_fea, left_index=True, right_index=True)

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

    # Simulation of bias for the PLAsTiCC dataset.
    # TODO: generalize this to more than just PLAsTiCC
    if args.simulate_plasticc_bias is not None:
        print("!!!!!")
        print("Simulating a bias in the redshift distribution of the "
              "augmented set by dropping Type Ia supernova observations!")

        # Modify the redshift distribution of the Type Ia supernovae.
        redshifts = dataset.metadata['host_specz']
        object_classes = dataset.metadata['class']

        if args.simulate_plasticc_bias == 'low_redshift':
            bias_thresholds = np.exp(-redshifts)
        elif args.simulate_plasticc_bias == 'high_redshift':
            bias_thresholds = np.exp(redshifts) - 1
        else:
            raise avocado.AvocadoException("Invalid bias type '%s'!" %
                                           args.simulate_plasticc_bias)

        bias_cut = np.random.rand(len(redshifts)) < bias_thresholds

        keep_mask = (
            (object_classes != 90)
            | dataset.metadata['reference_object_id'].isnull()
            | bias_cut
        )

        dataset.metadata = dataset.metadata[keep_mask]
        dataset.raw_features = dataset.raw_features[keep_mask]

        print("Dropped %d/%d objects!" % (np.sum(~keep_mask), len(keep_mask)))
        print("!!!!!")

    # drop classes
    classes_to_drop = None
    if args.only_var_stars:
        classes_to_drop = [16, 53, 92]
    elif args.only_supernova:
        classes_to_drop = [42, 52, 62, 64, 67, 90, 95]

    if classes_to_drop is not None:
        dataset.drop_classes(classes_to_drop)

    # Train the classifier
    print("Training classifier '%s'..." % args.classifier)
    classifier = LightGBMClassifier(
        "%s_%s" % (args.classifier, args.tag),
        MMMBOPFFeaturizer(include_metadata=args.use_metadata is not None),
        class_weights=class_weights,
        weighting_function=weighting_function,
        # settings_dir="predictions_mmmbopf_directory",
    )
    print("out path:", classifier.path, classifier)
    classifier.train(dataset)

    # Save the classifier
    print("Saving the classifier...")
    classifier.write(overwrite=True)

    print("Done!")
