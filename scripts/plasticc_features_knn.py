import argparse
import numpy as np

import avocado
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import pandas as pd
import time
from scipy import sparse

from src.preprocesing import gen_dataset, gen_dataset_from_h5
from src.feature_extraction.text import ParameterSelector, MPTextGenerator, TextGeneration, CountVectorizer
from src.feature_extraction.vector_space_model import VSM
from src.feature_extraction.centroid import CentroidClass
from src.feature_selection.select_k_best import SelectKTop
from src.decomposition import LSA, PCA
from src.neighbors import KNeighborsClassifier
from src.feature_extraction.window_slider import TwoWaysSlider

from sklearn.feature_selection import VarianceThreshold

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def cv_score(features, labels, classes, class_based, text='X'):
    ini = time.time()
    centroid = CentroidClass(classes=classes)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipeline = Pipeline([("centroid", centroid), ("knn", knn)])
    scores = cross_val_score(pipeline, features, labels, scoring="balanced_accuracy", cv=5, n_jobs=6, verbose=1)
    end = time.time()
    print("[%s]:" % text, np.mean(scores), "+-", np.std(scores), " (time: %.3f sec)" % (end - ini))

    y_pred = cross_val_predict(pipeline, features, labels, cv=5, n_jobs=6, verbose=1)
    return scores, pipeline, y_pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="YlGnBu"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
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

    # process features
    features = dataset.select_features(avocado.plasticc.PlasticcFeaturizer())

    # drop metadata from features
    features = features.drop(columns=["host_photoz", "host_photoz_error"])


    # drop nan columns
    nan_columns = features.columns[features.isnull().any()].tolist()
    print("nan_columns:", nan_columns)
    features = features.drop(columns=nan_columns)

    # transform to matrix
    features = features.to_numpy(dtype=float)

    # scale
    scale = StandardScaler()
    features = scale.fit_transform(features)

    # perform 1-NN classified on cross validation and using centroid classes
    print("Starting classifier...")
    scores, pipeline, y_pred = cv_score(features, labels, classes, True, text="avocado features 1NN")

    conf = confusion_matrix(labels, y_pred)
    fig = plt.figure(figsize=(10, 8))
    plot_confusion_matrix(conf, classes=classes, normalize=False,
                          title='Conf. matrix avocado features [b_acc:%.3f]' % balanced_accuracy_score(labels, y_pred))
    plt.savefig(os.path.join(main_path, "data", "conf_matrix_avocado_features.png", ), dpi=300)

    print("Done!")