# -*- coding: utf-8 -*-
from ..feature_selection.select_k_best import GeneralSelectKTop
from ..feature_selection.analysis_of_variance import manova_rank_fast
from ..feature_extraction.vector_space_model import VSM
from ..decomposition import LSA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import avocado
import os
import joblib
from .utils import check_file_path


def compact_method_pipeline(method, n_variables, n_features, classes):
    if method.C not in ["LSA", "lsa", "manova", "MANOVA"]:
        raise ValueError("invalid value for C={}".format(method.C))

    # get log-tf-idf
    vsm = VSM(class_based=False, classes=classes, norm=method.lsa_kw["normalize"],
              use_idf=method.lsa_kw["use_idf"], smooth_idf=True,
              sublinear_tf=method.lsa_kw["sublinear_tf"])

    # feature selection to k (gives matrix of shape (m, k, b))
    target_k = min(method.N // n_variables, n_features)
    # target_k = self.N // n_variables
    manova = GeneralSelectKTop(target_k, manova_rank_fast, allow_nd=True, n_variables=n_variables,
                               parameters_list=method.get_parameters())

    # latent semantic analysis
    target_k2 = min(method.N, int(n_features * n_variables))
    lsa = LSA(sc=target_k2, algorithm="randomized", n_iter=5, random_state=None, tol=0.)

    if method.C.lower() == "lsa":
        # print("LSA")
        pipeline = Pipeline([
            ("vsm", vsm),
            ("lsa", lsa),
        ])
        method.K = target_k2

    elif method.C.lower() == "manova":
        pipeline = Pipeline([
            ("manova", manova),
            ("vsm", vsm),
        ])
        method.K = target_k * n_variables
    else:
        raise ValueError("invalid value for C={}".format(method.C))
    return pipeline


class IBOPFPipelineProcessor(object):

    def __init__(self, filename, settings_dir, method_dir):
        self.pipeline = None
        self.settings_dir = settings_dir
        self.method_dir = method_dir
        self.filename = filename

    def get_working_dir(self):
        main_directory = avocado.settings[self.settings_dir]
        if not os.path.exists(main_directory):
            os.mkdir(main_directory)

        working_directory = os.path.join(main_directory, self.method_dir)
        if not os.path.exists(working_directory):
            os.mkdir(working_directory)

        return working_directory

    def save_pipeline(self, check_file=True, overwrite=True):
        working_directory = self.get_working_dir()
        if check_file:
            file = check_file_path(working_directory, self.filename, overwrite=overwrite)
        else:
            file = os.path.join(working_directory, self.filename)
        joblib.dump(self.pipeline, file)

    def load_pipeline(self):
        working_directory = self.get_working_dir()
        file = os.path.join(working_directory, self.filename)
        self.pipeline = joblib.load(file)

    def fit(self, x, y=None):
        self.pipeline = self.pipeline.fit(x, y=y)
        return self.pipeline

    def transform(self, x):
        return self.pipeline.transform(x)

    def fit_transform(self, x, y=None):
        return self.fit(x, y=y).transform(x)

    def set_pipeline(self, *args, **kwargs):
        pass


class CompactIBOPF(IBOPFPipelineProcessor):

    def __init__(self, filename="compact_pipeline.pkl", settings_dir="method_directory", method=None):
        super(CompactIBOPF, self).__init__(filename, settings_dir, "models")
        self.method = method if method is not None else "Unknown"

    def set_pipeline(self, method, n_variables, n_features, classes):
        self.pipeline = compact_method_pipeline(method, n_variables, n_features, classes)


class ZeroVarianceIBOPF(IBOPFPipelineProcessor):

    def __init__(self, filename="zero_variance_model.pkl", settings_dir="method_directory"):
        super(ZeroVarianceIBOPF, self).__init__(filename, settings_dir, "models")

    def set_pipeline(self, *args, **kwargs):
        self.pipeline = VarianceThreshold()