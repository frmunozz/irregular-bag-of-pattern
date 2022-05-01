import os
import pandas as pd
import numpy as np
import avocado
from avocado.classifier import Classifier as avocado_classifier
from abc import ABC
from ..neighbors import KNeighborsClassifier as knnclassifier
from ..feature_extraction.centroid import CentroidClass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm import tqdm
from .utils import get_classifier_path
from .featurizer import AVOCADOFeaturizer


class Classifier(avocado_classifier, ABC):
    def __init__(self, name, method="IBOPF"):
        super(Classifier, self).__init__(name)
        self.method = method

    @property
    def path(self):
        return get_classifier_path(self.name, method=self.method)


class LightGBMClassifier(avocado.LightGBMClassifier):

    def __init__(self,
                 name,
                 featurizer,
                 class_weights=None,
                 weighting_function=avocado.evaluate_weights_flat,
                 use_existing_features=False):

        super(LightGBMClassifier, self).__init__(name,
                                                 featurizer,
                                                 class_weights=class_weights,
                                                 weighting_function=weighting_function)
        method = "IBOPF"
        if isinstance(featurizer, AVOCADOFeaturizer):
            method = "AVOCADO"
        self.method = method
        self.use_existing_features = use_existing_features

    @property
    def path(self):
        return get_classifier_path(self.name, method=self.method)

    def write(self, overwrite=False):
        """Write a trained classifier to disk

        Parameters
        ----------
        name : str
            A unique name used to identify the classifier.
        overwrite : bool (optional)
            If a classifier with the same name already exists on disk and this
            is True, overwrite it. Otherwise, raise an AvocadoException.
        """
        import pickle

        path = self.path

        # Make the containing directory if it doesn't exist yet.
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        # Handle if the file already exists.
        if os.path.exists(path):
            if overwrite:
                avocado.logger.warning("Overwriting %s..." % path)
                os.remove(path)
            else:
                raise avocado.AvocadoException("Dataset %s already exists! Can't write." % path)

        # Write the classifier to a pickle file
        with open(path, "wb") as output_file:
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, name, method="IBOPF"):
        """Load a classifier that was previously saved to disk

        Parameters
        ----------
        name : str
            A unique name used to identify the classifier to load.
        """
        import pickle

        path = get_classifier_path(name, method=method)

        # Write the classifier to a pickle file
        with open(path, "rb") as input_file:
            classifier = pickle.load(input_file)

        return classifier

    def train(self, dataset, num_folds=None, random_state=None, **kwargs):
        """Train the classifier on a dataset

                Parameters
                ----------
                dataset : :class:`Dataset`
                    The dataset to use for training.
                num_folds : int (optional)
                    The number of folds to use. Default: settings['num_folds']
                random_state : int (optional)
                    The random number initializer to use for splitting the folds.
                    Default: settings['fold_random_state']
                **kwargs
                    Additional parameters to pass to the LightGBM classifier.
                """
        if dataset.features is not None and self.use_existing_features:
            features = dataset.features
        else:
            features = dataset.select_features(self.featurizer)

        # Label the folds
        folds = dataset.label_folds(num_folds, random_state)
        num_folds = np.max(folds) + 1

        object_weights = self.weighting_function(dataset, self.class_weights)
        object_classes = dataset.metadata["class"]
        classes = np.unique(object_classes)

        importances = pd.DataFrame()
        predictions = pd.DataFrame(
            -1 * np.ones((len(object_classes), len(classes))),
            index=dataset.metadata.index,
            columns=classes,
        )

        classifiers = []

        for fold in range(num_folds):
            print("Training fold %d." % fold)
            train_mask = folds != fold
            validation_mask = folds == fold

            train_features = features[train_mask]
            train_classes = object_classes[train_mask]
            train_weights = object_weights[train_mask]

            validation_features = features[validation_mask]
            validation_classes = object_classes[validation_mask]
            validation_weights = object_weights[validation_mask]

            classifier = avocado.fit_lightgbm_classifier(
                train_features,
                train_classes,
                train_weights,
                validation_features,
                validation_classes,
                validation_weights,
                **kwargs
            )

            validation_predictions = classifier.predict_proba(
                validation_features, num_iteration=classifier.best_iteration_
            )

            predictions[validation_mask] = validation_predictions

            importance = pd.DataFrame()
            importance["feature"] = features.columns
            importance["gain"] = classifier.feature_importances_
            importance["fold"] = fold + 1
            importances = pd.concat([importances, importance], axis=0, sort=False)

            classifiers.append(classifier)

        # Statistics on out-of-sample predictions
        total_logloss = avocado.weighted_multi_logloss(
            object_classes,
            predictions,
            object_weights=object_weights,
            class_weights=self.class_weights,
        )
        unweighted_total_logloss = avocado.weighted_multi_logloss(
            object_classes, predictions, class_weights=self.class_weights
        )
        print("Weighted log-loss:")
        print("    With object weights:    %.5f" % total_logloss)
        print("    Without object weights: %.5f" % unweighted_total_logloss)

        # Original sample only (no augments)
        if "reference_object_id" in dataset.metadata:
            original_mask = dataset.metadata["reference_object_id"].isnull()
            original_logloss = avocado.weighted_multi_logloss(
                object_classes[original_mask],
                predictions[original_mask],
                object_weights=object_weights[original_mask],
                class_weights=self.class_weights,
            )
            unweighted_original_logloss = avocado.weighted_multi_logloss(
                object_classes[original_mask],
                predictions[original_mask],
                class_weights=self.class_weights,
            )
            print("Original un-augmented dataset weighted log-loss:")
            print("    With object weights:    %.5f" % original_logloss)
            print("    Without object weights: %.5f" % unweighted_original_logloss)

        self.importances = importances
        self.train_predictions = predictions
        self.train_classes = object_classes
        self.classifiers = classifiers

        return classifiers

    def predict(self, dataset):
        """Generate predictions for a dataset

                Parameters
                ----------
                dataset : :class:`Dataset`
                    The dataset to generate predictions for.

                Returns
                -------
                predictions : :class:`pandas.DataFrame`
                    A pandas Series with the predictions for each class.
                """
        if dataset.features is not None and self.use_existing_features:
            features = dataset.features
        else:
            features = dataset.select_features(self.featurizer)

        predictions = 0

        for classifier in tqdm(self.classifiers, desc="Classifier", dynamic_ncols=True):
            fold_scores = classifier.predict_proba(
                features, raw_score=True, num_iteration=classifier.best_iteration_
            )

            exp_scores = np.exp(fold_scores)

            fold_predictions = exp_scores / np.sum(exp_scores, axis=1)[:, None]
            predictions += fold_predictions

        predictions /= len(self.classifiers)

        predictions = pd.DataFrame(
            predictions, index=features.index, columns=self.train_predictions.columns
        )

        return predictions


class KNNClassifier(Classifier):

    def __init__(self, name, featurizer, prototype=True, normalizer=False, scaler=False):
        if prototype:
            name += "_prototype"
        if normalizer:
            name += "_normalizer"
        if scaler:
            name += "_scaler"
        super(KNNClassifier, self).__init__(name)
        self.featurizer = featurizer
        self.prototype = prototype
        self.pipeline = None
        self.nan_columns = []
        self.normalizer = normalizer
        self.scaler = scaler

    def get_pipeline(self, labels):
        if self.prototype:
            # classes
            classes = np.unique(labels)

            centroid = CentroidClass(classes=classes)

            knn = knnclassifier(classes=classes, useClasses=True)

            pipeline = [("centroid", centroid), ("knn", knn)]

        else:

            norm = Normalizer()
            scale = StandardScaler()
            knn = knnclassifier(useClasses=False)
            pipeline = []
            if self.normalizer:
                pipeline.append(("normalize", norm))
            if self.scaler:
                pipeline.append(("scale", scale))
            pipeline.append(("knn", knn))

        return Pipeline(pipeline)

    def train(self, dataset):
        df_features = dataset.select_features(self.featurizer)
        self.nan_columns = df_features.columns[df_features.isna().any()].tolist()
        df_features = df_features.dropna(axis=1)
        # print("count:", df_features.isnull().sum())
        features = df_features.values

        labels = dataset.metadata["class"].to_numpy()

        pipeline = self.get_pipeline(labels)

        pipeline.fit(features, labels)

        self.pipeline = pipeline

        return pipeline

    def predict(self, dataset):

        df_features = dataset.select_features(self.featurizer)
        df_features = df_features.drop(self.nan_columns, axis=1)
        features = df_features.values

        pred_labels = self.pipeline.predict(features)

        object_id = dataset.metadata.index
        real_labels = dataset.metadata["class"]

        res_pd = pd.DataFrame({"class": real_labels, "pred": pred_labels}, index=dataset.metadata.index)

        return res_pd