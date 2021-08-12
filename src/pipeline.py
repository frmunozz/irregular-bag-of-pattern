import os
import pdb
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from collections import defaultdict
import time
from scipy import sparse

from src.preprocesing import gen_dataset_from_h5, label_folds, rearrange_splits
from src.feature_extraction.text import MPTextGeneratorMultivariateCountWords
from src.feature_selection.select_k_best import GeneralSelectKTop
from src.feature_selection.analysis_of_variance import manova_rank
from src.neighbors.knn import flattened_ddtw, Flatten3Dto2D
from sklearn.neighbors import KNeighborsClassifier
from src.neighbors import KNeighborsClassifier as knnclassifier
from src.feature_extraction.vector_space_model import VSM
from sklearn.preprocessing import Normalizer, StandardScaler
from src.decomposition import LSA, PCA
from src.feature_extraction.centroid import CentroidClass


symbols = {
    "mean": "M",
    "std": "S",
    "trend": "T",
    "min_max": "m",
}

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
_N_JOBS = 8


def quantities_code(quantities):
    f_name = ""
    for q_i in quantities:
        for q_ii in q_i:
            f_name += symbols[q_ii]
        f_name += "-"
    return f_name


class MMMBOPFPipeline(object):
    def __init__(self, alpha=4, Q=None, R=None, C=None,
                 lsa_kw=None, doc_kw=None, N=100,
                 max_dropped="default"):

        if C not in ["LSA", "lsa", "manova", "MANOVA"]:
            raise ValueError("invalid value for C={}".format(C))

        self.alpha = alpha
        self.Q = Q
        self.R = R
        self.lsa_kw = lsa_kw
        self.doc_kw = doc_kw
        self.N = N
        self.K = 0
        self.max_dropped = max_dropped
        self.C = C

    @property
    def n(self):
        return self.N

    def multi_variate_representation(self, data, win, wl, q):
        text_gen = MPTextGeneratorMultivariateCountWords(bands=_BANDS, n_jobs=_N_JOBS, win=win,
                                                         wl=wl, direct_bow=True, tol=wl * 2,
                                                         opt_desc=", " + "-".join(q), **self.doc_kw)
        return text_gen.fit_transform(data)

    def multi_quantity_representation(self, data, win, wl):
        if len(self.Q) > 1:
            data_repr_arr = []
            for q in self.Q:
                self.doc_kw["quantity"] = np.array(q)
                self.doc_kw["alphabet_size"] = np.array([self.alpha] * len(q))
                data_repr_i = self.multi_variate_representation(data, win, wl, q)
                data_repr_arr.append(data_repr_i)
            data_repr = sparse.hstack(data_repr_arr, format="csr")
        else:
            self.doc_kw["quantity"] = np.array(self.Q[0])
            self.doc_kw["alphabet_size"] = np.array([self.alpha] * len(self.Q[0]))
            data_repr = self.multi_variate_representation(data, win, wl, self.Q[0])

        return data_repr

    def cv_multi_resolution(self, data, labels, wins, wls, ):
        pass

    def compute_all_single_resolution(self, data, labels, wins, wls, cv=5):
        classes = np.unique(labels)
        q_code = quantities_code(self.Q)
        data_mr_repr = defaultdict(lambda: defaultdict(object))
        cv_results = defaultdict(lambda: defaultdict(object))
        result_lists = defaultdict(list)
        for wl in wls:
            for win in wins:
                message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
                try:
                    data_repr_i = self.multi_quantity_representation(data, win, wl)
                    data_mr_repr[win][wl] = data_repr_i
                    cv_results_i = self._repr_cv_score(data_repr_i, labels, classes, message=message, cv=cv)

                    if not (cv_results_i[0] is None and cv_results_i[1] is None):
                        cv_results[win][wl] = cv_results_i
                        result_lists["score"].append(np.mean(cv_results_i[0]))
                        result_lists["win"].append(win)
                        result_lists["wl"].append(wl)
                except Exception as e:
                    print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))
        return data_mr_repr, cv_results, result_lists

    def check_dropped(self, data_repr, labels):
        if self.max_dropped == "default":
            max_dropped = int(0.05 * len(labels))
        else:
            max_dropped = self.max_dropped

        dropped = len(np.where(np.sum(data_repr, axis=1) == 0)[0])
        return dropped,  max_dropped

    def get_sklearn_pipeline(self, n_variables, n_features, classes):
        # get log-tf-idf
        vsm = VSM(class_based=False, classes=classes, norm=self.lsa_kw["normalize"],
                  use_idf=self.lsa_kw["use_idf"], smooth_idf=True,
                  sublinear_tf=self.lsa_kw["sublinear_tf"])

        # feature selection to k (gives matrix of shape (m, k, b))
        target_k = min(self.N // n_variables, n_features)
        target_k = self.N // n_variables
        manova = GeneralSelectKTop(target_k, manova_rank, allow_nd=True, n_variables=n_variables)
        print("MANOVA TARGET K IS:", target_k)

        # flattening the matrix (gives matrix of shape (m, k*b))
        # flattening = Flatten3Dto2D()


        # normalize the resulting features
        normalize = Normalizer()

        # scaler
        scaler = StandardScaler()

        # latent semantic analysis
        target_k2 = min(self.N, int(n_features * n_variables))
        lsa = LSA(sc=target_k2, algorithm="randomized", n_iter=5, random_state=None, tol=0.)

        # centroid prototype
        centroid = CentroidClass(classes=classes)

        knn = knnclassifier(classes=classes, useClasses=True)
        knn2 = knnclassifier(classes=classes, useClasses=True, metric=flattened_ddtw,
                             metric_params={"shape": (self.n, n_variables)})
        # knn2: testing a different metric using DTW instead of euclidean distance

        if self.C.lower() == "lsa":
            print("USING LSA PIPELINE")
            pipeline = Pipeline([
                ("vsm", vsm),
                ("lsa", lsa),
                ("centroid", centroid),
                ("knn", knn),
            ])
            self.K = target_k2

        elif self.C.lower() == "manova":
            print("USING MANOVA PIPELINE")
            pipeline = Pipeline([
                ("manova", manova),
                ("vsm", vsm),
                ("centroid", centroid),
                ("knn", knn),
            ])
            self.K = target_k * n_variables
        else:
            raise ValueError("invalid value for C={}".format(self.C))
        return pipeline

    def quantities_code(self):
        f_name = ""
        for q_i in self.Q:
            for q_ii in q_i:
                f_name += symbols[q_ii]
            f_name += "-"
        return f_name

    def get_scheme_notation(self):
        scheme = "l" if self.lsa_kw["sublinear_tf"] else "n"
        scheme += "t" if self.lsa_kw["use_idf"] else "n"
        scheme += "c" if self.lsa_kw["normalize"] == "l2" else "n"
        return scheme
