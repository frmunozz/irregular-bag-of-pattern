import os
import pandas as pd
import numpy as np
import avocado
from avocado.classifier import Classifier as avocado_classifier
from avocado.features import Featurizer as avocado_featurizer
from abc import ABC
from scipy.special import erf
from .neighbors import KNeighborsClassifier as knnclassifier
from .feature_extraction.centroid import CentroidClass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from avocado.utils import logger
import time
from tqdm import tqdm
from .timeseries_object import TimeSeriesObject


settings = avocado.settings


class PlasticcAugmentor(avocado.plasticc.PlasticcAugmentor):

    def __init__(self):
        super(PlasticcAugmentor, self).__init__()
        self._min_detections = 2

    def augment_object(self, reference_object, force_success=True, custom_detections=True):
        if custom_detections:
            self._min_detections = np.sum(reference_object.observations["detected"])
        aug = super(PlasticcAugmentor, self).augment_object(reference_object, force_success=force_success)
        self._min_detections = 2
        return aug

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.
        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.
        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata
        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations["flux"]) / observations["flux_error"]
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        observations["detected"] = np.random.rand(len(s2n)) < prob_detected

        pass_detection = np.sum(observations["detected"]) >= self._min_detections

        return observations, pass_detection


class AVOCADOFeaturizer(avocado.plasticc.PlasticcFeaturizer):

    def __init__(self, discard_metadata=False, record_times=False):
        self.discard_metadata = discard_metadata
        self.record_times = record_times
        self.records = []

    def select_features(self, raw_features):

        features = super(AVOCADOFeaturizer, self).select_features(raw_features)

        if self.discard_metadata:
            features.pop("host_photoz")
            features.pop("host_photoz_error")

        return features

    def extract_raw_features(self, astronomical_object, return_model=False):
        ini = time.time()
        raw_features = super(AVOCADOFeaturizer, self).extract_raw_features(astronomical_object, return_model=return_model)
        end = time.time()

        if self.record_times:
            self.records.append([len(astronomical_object.observations), end - ini])

        return raw_features


class MMMBOPFFeaturizer(avocado.plasticc.PlasticcFeaturizer):

    def __init__(self, include_metadata=False, metadata_keys=None, method=None, zero_variance_model=None, compact_model=None):
        if metadata_keys is None:
            metadata_keys = ["host_photoz", "host_photoz_error"]

        self.metadata_keys = metadata_keys
        self.include_metadata = include_metadata
        self.metadata = None
        self.method = method
        self.zero_variance_model = zero_variance_model
        self.compact_model = compact_model

    def select_features(self, raw_features):
        # in this case raw features are the compact features
        # and we are going to append the metadata features
        if self.include_metadata:
            for k in self.metadata_keys:
                raw_features.loc[:, k] = self.metadata[k]

        return raw_features

    def extract_raw_features(self, astronomical_object, return_model=False):
        if self.method is None:
            raise ValueError("cannot run extraction without the method")

        data = TimeSeriesObject.from_astronomical_object(astronomical_object).fast_format_for_numba_code(astronomical_object.bands)

        sparse_data = self.method.mmm_bopf(data)

        if self.zero_variance_model is not None:
            sparse_data = self.zero_variance_model.transform(sparse_data)

        if self.compact_model is not None:
            compact_data = self.compact_model.transform(sparse_data)
            return compact_data
        else:
            return sparse_data


def get_classifier_path(name, settings_dir="classifier_directory"):
    """Get the path to where a classifier should be stored on disk

    Parameters
    ----------
    name : str
        The unique name for the classifier.
    """
    classifier_directory = settings[settings_dir]
    classifier_path = os.path.join(classifier_directory, "classifier_%s.pkl" % name)

    return classifier_path


class Classifier(avocado_classifier, ABC):
    def __init__(self, name, settings_dir="classifier_directory"):
        super(Classifier, self).__init__(name)
        self.settings_dir = settings_dir

    @property
    def path(self):
        return get_classifier_path(self.name, settings_dir=self.settings_dir)


class LightGBMClassifier(avocado.LightGBMClassifier):

    def __init__(self,
                 name,
                 featurizer,
                 class_weights=None,
                 weighting_function=avocado.evaluate_weights_flat,
                 settings_dir="classifier_directory"):

        super(LightGBMClassifier, self).__init__(name,
                                                 featurizer,
                                                 class_weights=class_weights,
                                                 weighting_function=weighting_function)

        self.settings_dir = settings_dir

    @property
    def path(self):
        return get_classifier_path(self.name, settings_dir=self.settings_dir)

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
        print("path:", path)

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
    def load(cls, name, settings_dir="classifier_directory"):
        """Load a classifier that was previously saved to disk

        Parameters
        ----------
        name : str
            A unique name used to identify the classifier to load.
        """
        import pickle

        path = get_classifier_path(name, settings_dir=settings_dir)

        # Write the classifier to a pickle file
        with open(path, "rb") as input_file:
            classifier = pickle.load(input_file)

        return classifier


class Dataset(avocado.Dataset):

    def __init__(self,
                 name,
                 metadata,
                 observations=None,
                 objects=None,
                 chunk=None,
                 num_chunks=None,
                 object_class=avocado.AstronomicalObject,
                 predictions_dir="predictions_directory"):

        super(Dataset, self).__init__(
            name,
            metadata,
            observations=observations,
            objects=objects,
            chunk=chunk,
            num_chunks=num_chunks,
            object_class=object_class)
        self.predictions_dir = predictions_dir
        self.records = None

    def get_predictions_path(self, classifier=None):

        if classifier is None:
            classifier = self.classifier

        if isinstance(classifier, str):
            classifier_name = classifier
        else:
            classifier_name = classifier.name

        filename = "predictions_%s_%s.h5" % (self.name, classifier_name)
        predictions_path = os.path.join(settings[self.predictions_dir], filename)

        return predictions_path

    def load_compact_features(self, features_tag, **kwargs):
        """Load the compact features from disk.

        Parameters
        ----------
        tag : str (optional)
            The version of the raw features to use. By default, this will use
            settings['features_tag'].

        Returns
        -------
        raw_features : pandas.DataFrame
            The extracted raw features.
        """
        features_directory = os.path.join(settings["method_directory"], "compact_features")

        # features_compact_LSA_plasticc_augment_v3
        features_filename = "%s_%s.h5" % (features_tag, self.name)
        features_path = os.path.join(features_directory, features_filename)

        self.raw_features = avocado.read_dataframe(
            features_path,
            "features",
            chunk=self.chunk,
            num_chunks=self.num_chunks,
            **kwargs
        )

        print("raw compact features shape:", self.raw_features.values.shape)

        return self.raw_features

    @classmethod
    def load(cls, name, metadata_only=False, chunk=None, num_chunks=None,
             object_class=avocado.AstronomicalObject, predictions_dir="predictions_directory",
             **kwargs):
        """Load a dataset that has been saved in HDF5 format in the data
                directory.

                For an example of how to create such a dataset, see
                `scripts/download_plasticc.py`.

                The dataset can optionally be loaded in chunks. To do this, pass chunk
                and num_chunks to this method. See `read_dataframes` for details.

                Parameters
                ----------
                name : str
                    The name of the dataset to load
                metadata_only : bool (optional)
                    If False (default), the observations are loaded. Otherwise, only
                    the metadata is loaded. This is useful for very large datasets.
                chunk : int (optional)
                    If set, load the dataset in chunks. chunk specifies the chunk
                    number to load. This is a zero-based index.
                num_chunks : int (optional)
                    The total number of chunks to use.
                **kwargs
                    Additional arguments to `read_dataframes`

                Returns
                -------
                dataset : :class:`Dataset`
                    The loaded dataset.
                """
        data_directory = settings["data_directory"]
        data_path = os.path.join(data_directory, name + ".h5")

        if not os.path.exists(data_path):
            raise avocado.AvocadoException("Couldn't find dataset %s!" % name)

        if metadata_only:
            keys = ["metadata"]
        else:
            keys = ["metadata", "observations"]

        dataframes = avocado.read_dataframes(
            data_path, keys, chunk=chunk, num_chunks=num_chunks, **kwargs
        )

        # Create a Dataset object
        dataset = cls(name, *dataframes, chunk=chunk, num_chunks=num_chunks,
                      object_class=object_class, predictions_dir=predictions_dir)

        return dataset

    def select_features(self, featurizer):
        """Select features from the dataset for classification.

        This method assumes that the raw features have already been extracted
        for this dataset and are available with `self.raw_features`. Use
        `extract_raw_features` to calculate these from the data directly, or
        `load_features` to recover features that were previously stored on
        disk.

        The features are saved as `self.features`.

        Parameters
        ----------
        featurizer : :class:`Featurizer`
            The featurizer that will be used to select the features.

        Returns
        -------
        features : pandas.DataFrame
            The selected features.
        """
        if self.raw_features is None:
            raise avocado.AvocadoException(
                "Must calculate raw features before selecting features!"
            )
        try:
            featurizer.metadata = self.metadata
        except Exception as e:
            pass

        features = featurizer.select_features(self.raw_features)

        self.features = features
        # print("FEATURES SELECTED SHAPE:", self.features.shape)

        return features

    def extract_raw_features(self, featurizer, keep_models=False):
        """(from AVOCADO)Extract raw features from the dataset.

                The raw features are saved as `self.raw_features`.

                Parameters
                ----------
                featurizer : :class:`AVOCADOFeaturizer`
                    The featurizer that will be used to calculate the features.
                keep_models : bool
                    If true, the models used for the features are kept and stored as
                    Dataset.models. Note that not all featurizers support this.

                Returns
                -------
                raw_features : pandas.DataFrame
                    The extracted raw features.
                """
        list_raw_features = []
        object_ids = []
        models = {}
        for obj in tqdm(self.objects, desc="Object", dynamic_ncols=True):
            obj_features = featurizer.extract_raw_features(
                obj, return_model=keep_models
            )

            if keep_models:
                obj_features, model = obj_features
                models[obj.metadata["object_id"]] = model

            list_raw_features.append(obj_features.values())
            object_ids.append(obj.metadata["object_id"])

        # Pull the keys off of the last extraction. They should be the same for
        # every set of features.
        keys = obj_features.keys()

        raw_features = pd.DataFrame(list_raw_features, index=object_ids, columns=keys)
        raw_features.index.name = "object_id"

        self.raw_features = raw_features

        if featurizer.record_times:
            records = pd.DataFrame(featurizer.records, index=object_ids, columns=["n", "time"])
            records.index.name = "object_id"
            self.records = records

        else:
            self.records = None

        if keep_models:
            self.models = models

        return raw_features

    def write_raw_features(self, tag=None, **kwargs):
        """(from AVOCADO)Write the raw features out to disk.

                The features will be stored in the features directory using the
                dataset's name and the given features tag.

                Parameters
                ----------
                tag : str (optional)
                    The tag for this version of the features. By default, this will use
                    settings['features_tag'].
                **kwargs
                    Additional arguments to be passed to `utils.write_dataframe`
                """
        raw_features_path = self.get_raw_features_path(tag=tag)

        avocado.write_dataframe(
            raw_features_path,
            self.raw_features,
            "raw_features",
            chunk=self.chunk,
            num_chunks=self.num_chunks,
            **kwargs
        )

        if self.records is not None:
            avocado.write_dataframe(
                raw_features_path,
                self.records,
                "record_times",
                chunk=self.chunk,
                num_chunks=self.num_chunks,
                **kwargs
            )


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
