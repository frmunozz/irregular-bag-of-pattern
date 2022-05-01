import os

import numpy as np

from .settings import get_path
from avocado.utils import logger, AvocadoException
from sklearn.neighbors import KNeighborsTransformer
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
from sklearn.preprocessing import StandardScaler


def get_neighbors_path(name, method="IBOPF"):
    """Get the path to where a similarity search based on nearest neighbors should be stored on disk

    Parameters
    ----------
    name : str
        The unique name for the classifier.
    """
    # classifier_directory = settings[method]["classifier_directory"]
    knn_ss_directory = get_path(method, "similarity_search_directory")
    knn_ss_path = os.path.join(knn_ss_directory, "knn_ss_%s.pkl" % name)

    return knn_ss_path


class KNNSimilaritySearch(object):

    def __init__(self, name, method="IBOPF", n_components=10, metric="cosine", n_jobs=-1, scale=True, with_mean=True):
        self.method = method
        self.name = name
        self.n_components = n_components
        self.metric = metric
        self.n_jobs = n_jobs
        self.fit_labels = None
        self.query_labels = None
        self.dists = None
        self.idxs = None
        self._knn = None
        self._scaler = None
        self._with_mean = with_mean
        self.scale = scale
        self._preprocess_pipeline = None
        self.k = self.n_components

    @property
    def path(self):
        """Get the path to where a classifier should be stored on disk"""
        return get_neighbors_path(self.name, method=self.method)

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
                logger.warning("Overwriting %s..." % path)
                os.remove(path)
            else:
                raise AvocadoException("Dataset %s already exists! Can't write." % path)

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

        path = get_neighbors_path(name, method=method)

        # Write the classifier to a pickle file
        with open(path, "rb") as input_file:
            classifier = pickle.load(input_file)

        return classifier

    def fit(self, x, y=None):
        if self._knn is None:
            self._knn = KNeighborsTransformer(n_neighbors=self.n_components, metric=self.metric, n_jobs=self.n_jobs)
        if self.scale:
            self._scaler = StandardScaler(with_mean=self._with_mean)
        if y is not None:
            try:
                self.fit_labels = y.to_numpy()
            except:
                if isinstance(y, list):
                    self.fit_labels = np.array(y)
                else:
                    self.fit_labels = y
        if self.scale:
            x = self._scaler.fit_transform(x)
        return self._knn.fit(x, y=y)

    def get_map_at_k(self, x, y=None, n_jobs=1, k=10):
        if k > self.n_components:
            raise ValueError("we cannot compute map@k for k higher than n_components")
        self.k = k
        if self.scale:
            x = self._scaler.transform(x)
        self.dists, self.idxs = self._knn.kneighbors(x)
        self.query_labels = y
        if n_jobs == 1:
            ap_list = []
            for i in range(len(x)):
                print(i, end="\r")
                ap_list.append(self.ap_worker(i))
            return ap_list
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()

            r = process_map(self.ap_worker, range(len(x)),
                            desc="[SIMILARITY SEARCH]", chunksize=1000)
            del self.dists
            del self.idxs
            del self.query_labels
            return r

    def ap_worker(self, query_idx):
        dists = self.dists[query_idx]
        fit_idxs = self.idxs[query_idx]
        fit_lbls = self.fit_labels[fit_idxs]
        query_lbl = self.query_labels[query_idx]

        sorted_idxs = np.argsort(dists)
        ap = 0
        true_counter = 0
        query_counter = 0
        for i in sorted_idxs:
            if query_counter > self.k:
                break
            query_counter += 1
            if query_lbl == fit_lbls[i]:
                true_counter += 1

            ap += true_counter / query_counter

        if true_counter == 0:
            ap = 0
        else:
            ap = ap / true_counter

        return ap


