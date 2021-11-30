import argparse
import numpy as np

import avocado
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, main_path)
from sklearn.preprocessing import StandardScaler

from src.feature_extraction.centroid import CentroidClass
from src.neighbors import KNeighborsClassifier

from tqdm import tqdm

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def train_classifier(features, labels, classes, class_based):
    centroid = CentroidClass(classes=classes)
    f2 = centroid.fit_transform(features, y=labels)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    knn.fit(f2, classes)
    return knn


def process_chunk(classifier, scaler, nan_columns, chunk, args):

    test_set = avocado.load(args.test_set, metadata_only=True, chunk=chunk,
                            num_chunks=args.num_chunks)
    test_set.load_raw_features()
    test_labels = test_set.metadata['class'].to_numpy()
    df = test_set.raw_features

    # drop nan
    df = df.drop(columns=nan_columns)

    # get matrix
    test_X = df.to_numpy(dtype=float)

    #scale
    test_X = scaler.transform(test_X)

    # predict
    predictions = classifier.predict(test_X)

    # write dataframe
    out_df = test_set.metadata
    out_df["predict"] = predictions
    # out_df = pd.DataFrame(out_dict)
    out_filename = "knn_test_results_plasticc.h5"
    out_path = os.path.join(avocado.settings["knn_directory"], out_filename)

    avocado.write_dataframe(out_path, out_df, "knn",
                            chunk=chunk, num_chunks=args.num_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        '--test_set',
        help='specify test set for knn',
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

    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset '%s'..." % args.dataset)
    dataset = avocado.load(args.dataset, metadata_only=True)
    labels = dataset.metadata['class'].to_numpy()
    classes = np.unique(labels)

    # Load the dataset raw features
    print("Loading raw features...")
    dataset.load_raw_features()
    df = dataset.raw_features

    print("Preprocessing raw features...")

    # drop nan columns
    nan_columns = df.columns[df.isnull().any()].tolist()
    df = df.drop(columns=nan_columns)

    # transform to matrix
    features = df.to_numpy(dtype=float)

    # scale
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # train classifier
    print("train knn classifier...")
    classifier = train_classifier(features, labels, classes, True)

    # predict in chunks
    if args.chunk is not None:
        # Process a single chunk
        process_chunk(classifier, scaler, nan_columns, args.chunk, args)
    else:
        # Process all chunks
        print("Processing the dataset in %d chunks..." % args.num_chunks)
        for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                          dynamic_ncols=True):
            process_chunk(classifier, scaler, nan_columns, chunk, args)

    print("Done!")