import pickle
import json
import numpy as np
from ..feature_extraction.text.optimal_text_generation import mp_text_transform, generate_multivariate_text, get_alpha_by_quantity, multiprocessing_extended_ibopf
from ..feature_selection.select_k_best import GeneralSelectKTop
from ..feature_selection.analysis_of_variance import manova_rank_fast
from ..neighbors import KNeighborsClassifier as knnclassifier
from ..feature_extraction.vector_space_model import VSM
from ..decomposition import LSA
from ..feature_extraction.centroid import CentroidClass
from ibopf.utils import AbstractCore
from .models import compact_method_pipeline
from .utils import quantity_code_extend, _SYMBOLS
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


class IBOPF(object):
    def __init__(self,
                 alpha=4,
                 Q=None,
                 R=None,
                 C=None,
                 Q_code=None,
                 lsa_kw=None,
                 doc_kw=None,
                 N=100,
                 max_dropped="default",
                 n_jobs=-1,
                 drop_zero_variance=False,
                 bands=None):

        if Q is None and Q_code is not None:
            Q = quantity_code_extend(Q_code)

        if bands is None:
            # use default bands for plasticc
            bands = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]

        self.alpha = alpha
        self.Q = Q
        self.R = R
        self.lsa_kw = lsa_kw
        self.doc_kw = doc_kw
        self.N = N
        self.K = 0
        self.max_dropped = max_dropped
        self.C = C
        self.n_jobs = n_jobs
        self.drop_zero_variance = drop_zero_variance
        self.model = None
        self.print_ssm_time = False
        self.bands = bands

    @property
    def n(self):
        return self.N

    def ssm_bopf(self, data, win, wl, chunk=None, record_times=False):
        q_symbols = self.quantities_code()
        opt_desc = ", %s" % q_symbols
        if chunk is not None:
            opt_desc += ", chunk: %d" % chunk

        # TODO: this call has yet to be optimized
        text_res, elapse, time_records, length_records, widths_records = mp_text_transform(
            data, len(self.bands), win=win, wl=wl,
            alphabet_size=self.doc_kw["alphabet_size"],
            quantity=self.doc_kw["quantity"],
            quantity_symbol=self.quantities_code([self.doc_kw["quantity"]]),
            num_reduction=True, tol=wl * 2,
            mean_bp_dist=self.doc_kw["mean_bp_dist"],
            threshold=None, n_jobs=self.n_jobs, print_time=self.print_ssm_time, record_times=record_times)

        return text_res, elapse, time_records, length_records

    def smm_bopf(self, data, win, wl, chunk=None, record_times=False):
        # print("len self.Q", len(self.Q))
        if len(self.Q) > 1:
            data_repr_arr = []
            elapse = 0
            time_records = None
            length_records = None
            for q in self.Q:
                if isinstance(q, str):
                    q = [q]
                self.doc_kw["quantity"] = np.array(q)
                self.doc_kw["alphabet_size"] = np.array([self.alpha] * len(q))
                data_repr_i, elapse_i, time_records_ssm, length_records_ssm = self.ssm_bopf(data, win, wl, chunk=chunk, record_times=record_times)
                if record_times:
                    if length_records is None:
                        length_records = length_records_ssm
                        time_records = time_records_ssm
                    else:
                        time_records += time_records_ssm
                data_repr_arr.append(data_repr_i)
                elapse += elapse_i
            data_repr = sparse.hstack(data_repr_arr, format="csr")
        else:
            self.doc_kw["quantity"] = np.array(self.Q[0])
            self.doc_kw["alphabet_size"] = np.array([self.alpha] * len(self.Q[0]))
            data_repr, elapse, time_records, length_records = self.ssm_bopf(data, win, wl, chunk=chunk)

        return data_repr, elapse, time_records, length_records

    def mmm_bopf(self, data, R=None, chunk=None, record_times=False):
        if R is not None:
            self.R = R
        data_repr = []
        time_records = None
        length_records = None
        for pair in self.R:
            win, wl = pair
            repr_pair, elapse, time_records_smm, lenght_records_smm = self.smm_bopf(data, win, wl, chunk=chunk, record_times=record_times)
            # print("shape:", repr_pair.shape)
            if record_times:
                if length_records is None:
                    length_records = lenght_records_smm
                    time_records = time_records_smm
                else:
                    time_records += time_records_smm
            data_repr.append(repr_pair)
        x = sparse.hstack(data_repr, format="csr")

        if record_times:
            return x, time_records, length_records

        return x

    def get_parameters(self):
        parameters = []
        for pair in self.R:
            win, wl = pair
            for q in self.Q:
                if isinstance(q, str):
                    q = [q]

                quantity = q
                alphabet_size = [self.alpha] * len(q)
                quantity_symbol = self.quantities_code([quantity])
                tol = wl * 2
                mean_bp_dist = self.doc_kw["mean_bp_dist"]
                num_reduction = True
                threshold = wl
                # (win, wl, q, alpha, q_symbol, tol, mean_bp, num_reduction, threshold)
                parameters.append((win, wl, quantity, alphabet_size, quantity_symbol,
                                   tol, mean_bp_dist, num_reduction, threshold))
        return parameters

    def extended_IBOPF(self, data, record_times=False, verbose=False):
        if verbose:
            self.print_config()
        parameters = self.get_parameters()

        features, records = multiprocessing_extended_ibopf(
            data, parameters, len(self.bands), n_jobs=self.n_jobs,
            print_time=self.print_ssm_time, record_times=record_times)

        if record_times:
            return features, records
        return features

    def check_dropped(self, data_repr, labels):
        if self.max_dropped == "default":
            max_dropped = int(0.05 * len(labels))
        else:
            max_dropped = self.max_dropped

        dropped = len(np.where(np.sum(data_repr, axis=1) == 0)[0])
        return dropped,  max_dropped

    def get_compact_pipeline(self, n_variables, n_features, classes):
        if self.C not in ["LSA", "lsa", "manova", "MANOVA"]:
            raise ValueError("invalid value for C={}".format(self.C))

        # get log-tf-idf
        vsm = VSM(class_based=False, classes=classes, norm=self.lsa_kw["normalize"],
                  use_idf=self.lsa_kw["use_idf"], smooth_idf=True,
                  sublinear_tf=self.lsa_kw["sublinear_tf"])

        # feature selection to k (gives matrix of shape (m, k, b))
        target_k = min(self.N // n_variables, n_features)
        # target_k = self.N // n_variables
        manova = GeneralSelectKTop(target_k, manova_rank_fast, allow_nd=True, n_variables=n_variables)

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

        if self.C.lower() == "lsa":
            # print("LSA")
            pipeline = Pipeline([
                ("vsm", vsm),
                ("lsa", lsa),
                ("centroid", centroid),
                ("knn", knn),
            ])
            self.K = target_k2

        elif self.C.lower() == "manova":
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

    def get_compact_method_pipeline(self, n_variables, n_features, classes):
        return compact_method_pipeline(self, n_variables, n_features, classes)

    def quantities_code(self, Q=None):
        if Q is None:
            Q = self.Q
        f_arr = []
        for q_i in Q:
            if isinstance(q_i, str):
                f_arr.append(_SYMBOLS[q_i])
            else:
                ff_arr = []
                for q_ii in q_i:
                    ff_arr.append(_SYMBOLS[q_ii])
                ff_name = "".join(ff_arr)
                f_arr.append(ff_name)
        f_name = "(" + "-".join(f_arr) + ")"

        return f_name

    def get_scheme_notation(self):
        scheme = "l" if self.lsa_kw["sublinear_tf"] else "n"
        scheme += "t" if self.lsa_kw["use_idf"] else "n"
        scheme += "c" if self.lsa_kw["normalize"] == "l2" else "n"
        return scheme

    def config_to_json(self, filename):
        print("config saving data to json:")
        self.doc_kw["quantity"] = []
        self.doc_kw["alphabet_size"] = []

        config = {
            "alpha": self.alpha,
            "Q": list(self.Q),
            "R": list(self.R),
            "docKwargs": self.doc_kw,
            "lsaKwargs": self.lsa_kw,
            "N": self.N,
            "K": self.K,
            "maxDropped": self.max_dropped,
            "C": self.C,
            "dropZeroVariance": self.drop_zero_variance
        }
        a_file = open(filename, "w")
        json.dump(config, a_file)

    def config_from_json(self, filename):
        a_file = open(filename, "r")
        config = json.load(a_file)
        self.alpha = config["alpha"]
        self.Q = config["Q"]
        self.R = config["R"]
        self.lsa_kw = config["lsaKwargs"]
        self.doc_kw = config["docKwargs"]
        self.N = config["N"]
        self.K = config["K"]
        self.max_dropped = config["maxDropped"]
        self.C = config["C"]
        self.drop_zero_variance = config["dropZeroVariance"]

    def print_config(self):
        print("CONFIG FOR GENERAL IBOPF:")
        print(">> ALPHA: ", self.alpha)
        print(">> STATISTICAL QUANTITIES: ", self.quantities_code(self.Q))
        print(">> LEVELS OF RESOLUTION: ", self.R)
        print(">> COMPACT METHOD: ", self.C)
        print(">> TARGET NUMBER OF COMPONENTS: ", self.N)