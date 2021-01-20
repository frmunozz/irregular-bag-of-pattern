import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from ...utils import get_vocabulary_size
from scipy import sparse


def count_words(corpus, vocabulary_size):
    n = len(corpus)
    bow = sparse.lil_matrix((n, vocabulary_size), dtype=int)

    for i in range(n):
        doc = corpus[i]
        if doc is not None:
            for word in doc:
                bow[i, word] += 1
    return sparse.csr_matrix(bow)


def merge_documents(corpus, bands, vocabulary_size):
    new_corpus = np.full(corpus.shape, None, dtype=object)
    for i in range(len(corpus)):
        if corpus[i] is not None:
            merged_doc = np.array([], dtype=int)
            for j, b in enumerate(bands):
                merged_doc = np.append(merged_doc,
                                    np.array(corpus[i][b], dtype=int) + vocabulary_size * j,)
            new_corpus[i] = merged_doc
    return new_corpus


class Vectorizer(TransformerMixin, BaseEstimator):

    def __init__(self, **kwargs):
        self.bop_size = get_vocabulary_size(kwargs.get("alph_size"), kwargs.get("word_length"),
                                            kwargs.get("irr_handler"))
        self.bands = kwargs.get("bands")

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        new_x = merge_documents(X, self.bands, self.bop_size)
        return count_words(new_x, self.bop_size * len(self.bands))


class MRVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, alph_size=None, word_length=None, empty_handler=None, bands=None):
        self.bop_size = [get_vocabulary_size(a, w, empty_handler) for a, w in zip(alph_size, word_length)]
        self.bands = bands
