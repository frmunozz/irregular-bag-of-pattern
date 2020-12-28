from src.Adeprecated.transformer.document_transformer import DocumentGeneration
from src.Adeprecated.transformer import VectorSpaceModel
from ..reduction.anova import ANOVA
from ..reduction.LSA import LSA
from src.dataset import Dataset


class IBOPFPipeline:
    def __init__(self):
        self.corpus = None
        self.feature_matrix = None
        self.failed = 0
        self.vocabulary_size = 0

    def generate_documents(self, win, wl, dataset):
        doc_gen = DocumentGeneration(win, word_length=wl, alph_size=4, quantity="mean")
        self.corpus, self.failed = doc_gen.transform_dataset(dataset)
        self.vocabulary_size = doc_gen.bop_size

    # def merge_multi_documents(self):


    def compute_count_words(self):
        self.count_words(self.corpus, self.vocabulary_size)


def horizontal_stack_matrix(feature_matrix):
    return feature_matrix


def compute_centroid(feature_matrix, labels):
    return feature_matrix


def merge_by_class(feature_matrix, labels):
    return feature_matrix, labels


class IrregularBOPF(object):

    def __init__(self, win, wl, k, class_type=None, scheme="log-tf-idf", reducer="ANOVA"):
        # self._cw_transf = CountWords(win, wl)
        self._vsm_transf = VectorSpaceModel(scheme)
        if reducer.upper() == "ANOVA":
            self._red_transf = ANOVA(k, class_type=class_type)
        else:
            self._red_transf = LSA(k)
        self._class_type = class_type

    def transform_train_set(self, d: Dataset):
        labels = d.labels
        feature_matrix = self._cw_transf.transform(d.time_series)
        feature_matrix = horizontal_stack_matrix(feature_matrix)

        if self._class_type == "type-1":
            feature_matrix, labels = merge_by_class(feature_matrix, labels)

        self._vsm_transf.fit(feature_matrix)
        feature_matrix = self._vsm_transf.transform(feature_matrix)

        self._red_transf.fit(feature_matrix, labels)
        feature_matrix = self._red_transf.transform(feature_matrix)

        if self._class_type == "type-2":
            feature_matrix = compute_centroid(feature_matrix, labels)

        return feature_matrix

    def transform_test_set(self, d: Dataset):
        feature_matrix = self._cw_transf.transform(d.time_series)
        feature_matrix = horizontal_stack_matrix(feature_matrix)

        feature_matrix = self._vsm_transf.transform(feature_matrix)

        feature_matrix = self._red_transf.transform(feature_matrix)

        return feature_matrix

