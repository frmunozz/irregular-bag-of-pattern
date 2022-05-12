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
import umap
from ..decomposition.pacmap import PaCMAP
from ..settings import settings, get_path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm.contrib.concurrent import process_map


def compact_method_pipeline(method, n_variables, n_features, classes, n_neighbors=15,
                            min_dist=0.0, metric="hellinger", densmap=False, sparse_split=None):
    if method.C not in ["LSA", "lsa", "manova", "MANOVA", "UMAP", "PACMAP"]:
        raise ValueError("invalid value for C={}".format(method.C))

    # get log-tf-idf
    vsm = VSM(class_based=False, classes=classes, norm=method.lsa_kw["normalize"],
              use_idf=method.lsa_kw["use_idf"], smooth_idf=True, return_dense=method.C.lower() in ["umap", "pacmap"],
              sublinear_tf=method.lsa_kw["sublinear_tf"], sparse_split=sparse_split)

    # scaler
    scaler = StandardScaler(with_mean=False)

    if method.C.lower() == "lsa":
        # latent semantic analysis
        target_k2 = min(method.N, int(n_features * n_variables))
        lsa = LSA(sc=target_k2, algorithm="randomized", n_iter=5, random_state=None, tol=0.)
        # print("LSA")
        pipeline = Pipeline([
            ("vsm", vsm),
            ("scaler", scaler),
            ("lsa", lsa),
        ])
        method.K = target_k2

    elif method.C.lower() == "manova":
        # feature selection to k (gives matrix of shape (m, k, b))
        target_k = min(method.N // n_variables, n_features)
        # target_k = self.N // n_variables
        manova = GeneralSelectKTop(target_k, manova_rank_fast, allow_nd=True, n_variables=n_variables,
                               parameters_list=method.get_parameters())
        pipeline = Pipeline([
            ("manova", manova),
            ("vsm", vsm),
        ])
        method.K = target_k * n_variables
    elif method.C.lower() == "umap":
        # umap
        # comentarios de exploraciones a realizar
        # las variables a probar son:
        # - min_dist: 0, 0.1, 0.5 -> verificar si mas o menos disperso ayuda
        #### a menor distancia mejor, se utiliza dist 0.0
        # - n neighbors: 50 100 150 -> verificar como influyen, es lo ultimo a testar, iniciar con 50
        #### resultados preliminares mostraron que a mas n neighbor mayor structura globa y mejores resultados
        #### pero es mas lento igual, trade-off
        #### lo dejaremos en 100 y sin optimizar
        # - distance metric: euclidean, cosine, hellinger -> verificar diferentes casos
        ##### hellinger was best
        # - supervised / unsupervised
        ##### resultados preliminares indican que el set de entrenamiento no es suficientemente representativo
        ##### y por tanto, es mejor utilizar unsupervised

        # - probar la topologia, quizas separar con super-nova y el resto para ver si aprende un mejor
        #                       espacio para las supernoba. Luego intersectar o unir, verificar ambos
        # - mutual graph k-NN, opciones son:
        #       * NN + adjacent neighbors
        #       * MST-MIN + adjacent neighbors
        #       * MST-all + adjacent neighbors
        #       * MST-MIN + PATH NEIGHBORS
        # - UTILIZAR DENSMAP -> probar despues de optimizar todo lo anterior
        #### muy lento y requiere de mucha ram, no se utilizara
        # - parametric umap -> utiliza encoder-decoder!
        target_k2 = min(method.N, int(n_features * n_variables))
        umap_reducer = umap.UMAP(n_components=target_k2-1, n_neighbors=n_neighbors, verbose=True, random_state=42, 
                            min_dist=min_dist, metric=metric, densmap=densmap, n_jobs=-1)
        pipeline = Pipeline([
            ("vsm", vsm),
            ("scaler", scaler),
            ("umap", umap_reducer),
        ])
        method.K = target_k2
    elif method.C.lower() == "pacmap":
        # pacmap
        target_k2 = min(method.N, int(n_features * n_variables))
        pacmap_reducer = PaCMAP(n_components=target_k2-1, n_neighbors=n_neighbors, MN_ratio=0.5, FP_ratio=4.0, verbose=True,
                                   random_state=42, save_tree=True)
        pipeline = Pipeline([
            ("vsm", vsm),
            ("scaler", scaler),
            ("pacmap", pacmap_reducer),
        ])
        method.K = target_k2
    else:
        raise ValueError("invalid value for C={}".format(method.C))

    print("target n_componets %d, n neighbors %d" % (method.K-1, n_neighbors))
    return pipeline


class IBOPFPipelineProcessor(object):

    def __init__(self, filename):
        self.pipeline = None
        self.filename = filename
        self.basis = None

    def get_working_dir(self):
        main_directory = get_path("IBOPF", "directory")
        # settings["IBOPF"][self.settings_dir]
        if not os.path.exists(main_directory):
            os.mkdir(main_directory)

        working_directory = get_path("IBOPF", "models_directory")
        # working_directory = os.path.join(main_directory, self.method_dir)
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

    def transform(self, x, basis=None, concurrent=False):
        self.basis = basis
        if concurrent:
            r = process_map(self._transform_worker, x, max_workers=4,
                        desc="[Transform]", chunksize=1000)
        else:
            r = self._transform_worker(x)

        return r

    def _transform_worker(self, x):
        if self.basis is not None:
            return self.pipeline.transform(x, self.basis)
        else:
            return self.pipeline.transform(x)


    def fit_transform(self, x, y=None):
        return self.fit(x, y=y).transform(x)

    def grid_search_fit(self, X, y=None, method="umap"):
        param_grid = {
            "%s__n_components" % method: [2, 5, 15, 30],
        }
        search = GridSearchCV(self.pipeline, param_grid, n_jobs=-1, verbose=1, cv=5)
        search.fit(X, y=y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        self.pipeline = search
        return self.pipeline


    def set_pipeline(self, *args, **kwargs):
        pass


class CompactIBOPF(IBOPFPipelineProcessor):

    def __init__(self, filename="compact_pipeline.pkl", settings_dir="directory", method=None):
        super(CompactIBOPF, self).__init__(filename)
        self.method = method if method is not None else "Unknown"

    def set_pipeline(self, method, n_variables, n_features, classes, n_neighbors=15,
                     min_dist=0.0, metric="hellinger", densmap=False, sparse_split=None):
        self.pipeline = compact_method_pipeline(method, n_variables, n_features, classes, 
            n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, densmap=densmap, sparse_split=sparse_split)


class ZeroVarianceIBOPF(IBOPFPipelineProcessor):

    def __init__(self, filename="zero_variance_model.pkl", settings_dir="directory"):
        super(ZeroVarianceIBOPF, self).__init__(filename)

    def set_pipeline(self, *args, **kwargs):
        self.pipeline = VarianceThreshold()