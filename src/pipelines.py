from .feature_extraction.text import ParameterSelector, MPTextGenerator, CountVectorizer
from .feature_extraction.vector_space_model import VSM
from .feature_extraction.centroid import CentroidClass
from .feature_selection.select_k_best import SelectKBest
from .decomposition.lsa import LSA

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


class PipelineBuilder:

    def __init__(self, class_based=False, class_type="type-1", classes=None):
        """

        :param class_based:
        :param class_type:
        """
        self.pipe = []
        self.class_type = class_type
        self.class_based = class_based
        self.classes = classes

    def set_feature_extraction(self, precomputed=True, **kwargs):
        """

        :param precomputed:
        :param kwargs:
        :return:
        """
        if precomputed:
            self.pipe.append(("ext", ParameterSelector(**kwargs)))
        else:
            self.pipe.append(("ext", MPTextGenerator(**kwargs)))
            self.pipe.append(("vec", CountVectorizer(**kwargs)))

    def set_transformer(self, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True):
        """

        :param norm:
        :param use_idf:
        :param smooth_idf:
        :param sublinear_tf:
        :return:
        """
        class_based = self.class_based and self.class_type == "type-1"
        self.pipe.append(("vsm", VSM(class_based=class_based, classes=self.classes, norm=norm, use_idf=use_idf,
                                     smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)))

    def set_reducer(self, spatial_complexity, reducer_type="lsa",
                    algorithm="randomized", n_iter=5,
                    random_state=None, tol=0.):
        """

        :param spatial_complexity:
        :param reducer_type:
        :param algorithm:
        :param n_iter:
        :param random_state:
        :param tol:
        :return:
        """
        class_based_1 = self.class_based and self.class_type == "type-1"
        class_based_2 = self.class_based and self.class_type == "type-2"
        if class_based_1:
            red = LSA(spatial_complexity, algorithm=algorithm, n_iter=n_iter, random_state=random_state, tol=tol)
        else:
            if reducer_type.lower() == "lsa":
                red = LSA(spatial_complexity, algorithm=algorithm, n_iter=n_iter, random_state=random_state, tol=tol)
            elif reducer_type.lower() == "anova":
                red = SelectKBest(spatial_complexity)
            else:
                raise ValueError("reducer '%s' not implemented" % reducer_type)

        self.pipe.append(("red", red))

        if class_based_2:
            self.pipe.append(("centroid", CentroidClass(classes=self.classes)))

    def set_normalizer(self):
        """

        :return:
        """
        self.pipe.append(("norm", Normalizer()))

    def set_feature_scaler(self, with_mean=True, with_std=True):
        """

        :param with_mean:
        :param with_std:
        :return:
        """
        self.pipe.append(("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)))

    def set_classifier(self, n_neighbors=1):
        """

        :param n_neighbors:
        :return:
        """
        self.pipe.append(("classif", KNeighborsClassifier(n_neighbors=n_neighbors)))

    def build(self):
        """

        :return:
        """
        # validate good pipe
        names = []
        for step in self.pipe:
            n = step[0]
            if n not in names:
                names.append(n)
            else:
                raise ValueError("bad pipeline, step '%s' is repeated".format(n))

        return Pipeline(self.pipe)
