import os
import pandas as pd
import avocado
from tqdm import tqdm
from ..settings import settings, get_path, get_data_directory
import os


class Dataset(avocado.Dataset):

    def __init__(self,
                 name,
                 metadata,
                 observations=None,
                 objects=None,
                 chunk=None,
                 num_chunks=None,
                 object_class=avocado.AstronomicalObject):

        super(Dataset, self).__init__(
            name,
            metadata,
            observations=observations,
            objects=objects,
            chunk=chunk,
            num_chunks=num_chunks,
            object_class=object_class)
        self.method = None
        self.records = None
        self.sparse_features = None

    def set_method(self, method):
        self.method = method

    @property
    def path(self):
        """Return the path to where this dataset should lie on disk"""
        # data_directory = settings["data_directory"]
        data_directory = get_data_directory()
        data_path = os.path.join(data_directory, self.name + ".h5")

        return data_path

    def get_raw_features_path(self, tag=None, dir_key="features_directory"):
        """(copy from avocado to update new settings)
        Return the path to where the raw features for this dataset should
        lie on disk

        Parameters
        ----------
        tag : str (optional)
            The version of the raw features to use. By default, this will use
            settings['features_tag'].
        dir_key : str (optional)
        """
        if tag is None:
            tag = settings["features_tag"]

        # features_directory = settings[self.method][dir_key]
        features_directory = get_path(self.method, dir_key)

        features_filename = "%s_%s.h5" % (tag, self.name)
        features_path = os.path.join(features_directory, features_filename)

        return features_path

    def get_models_path(self, tag=None):
        """(copy from avocado to update new settings)
        Return the path to where the models for this dataset should lie on
        disk

        Parameters
        ----------
        tag : str (optional)
            The version of the features/model to use. By default, this will use
            settings['features_tag'].
        """
        if tag is None:
            tag = settings[self.method]["features_tag"]

        features_directory = settings[self.method]["features_directory"]
        # features_directory = get_path(self.method, "features_directory")

        models_filename = "models_%s_%s.h5" % (tag, self.name)
        models_path = os.path.join(features_directory, models_filename)

        return models_path

    def get_predictions_path(self, classifier=None):

        if classifier is None:
            classifier = self.classifier

        if isinstance(classifier, str):
            classifier_name = classifier
        else:
            classifier_name = classifier.name

        filename = "predictions_%s_%s.h5" % (self.name, classifier_name)
        # predictions_path = os.path.join(settings[self.method]["predictions_directory"], filename)
        predictions_path = os.path.join(get_path(self.method, "predictions_directory"), filename)

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
        # features_directory = settings[self.method]["compact_features_directory"]
        features_directory = get_path(self.method, "compact_features_directory")


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

        # print("raw compact features shape:", self.raw_features.values.shape)

        return self.raw_features

    def drop_classes(self, classes):
        self.metadata = self.metadata[~self.metadata["class"].isin(classes)]
        self.raw_features = self.raw_features[self.raw_features.index.isin(self.metadata.index)]

    def load_sparse_features(self, features_tag, **kwargs):
        # features_directory = settings[self.method]["sparse_features_directory"]
        features_directory = get_path(self.method, "sparse_features_directory")

        # features_v3_LSA_plasticc_augment_v3.h5
        features_filename = "%s_%s.h5" % (features_tag, self.name)
        features_path = os.path.join(features_directory, features_filename)

        self.raw_features = avocado.read_dataframe(
            features_path,
            "features",
            chunk=self.chunk,
            num_chunks=self.num_chunks,
            **kwargs
        )

        print("raw sparse features shape:", self.raw_features.values.shape)

        return self.raw_features


    @classmethod
    def load(cls, name, metadata_only=False, chunk=None, num_chunks=None,
             object_class=avocado.AstronomicalObject,
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
        # data_directory = settings["data_directory"]
        data_directory = get_data_directory()
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
                      object_class=object_class)

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
        metadata = self.metadata

        if self.metadata.shape[0] != self.raw_features.shape[0]:
            # print("reducing metadata")
            metadata = self.metadata[self.metadata.index.isin(self.raw_features.index)]

        try:
            featurizer.metadata = metadata
        except Exception as e:
            print("failed to set metadata on featurizer, error: %s" % e)


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