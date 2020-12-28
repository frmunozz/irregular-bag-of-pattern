from src.Adeprecated.sax import Alphabet, SymbolicAggregateApproximation, PiecewiseAggregateApproximation
import numpy as np


class BagOfPatternFeature(object):
    def __init__(self, n, alphabet=None, strategy_paa="time_window", strategy_sax="normal"):
        if alphabet is None:
            alphabet = Alphabet(['a', 'b', 'c', 'd'])
        self.alphabet = alphabet
        self.n = n
        self.strategy_paa = strategy_paa
        self.strategy_sax = strategy_sax
        self.tf = []
        self.idf = {}
        self.doc_contain_words = self.alphabet.get_count_words_dict()
        self.paa = PiecewiseAggregateApproximation(self.n, strategy=self.strategy_paa)
        self.sax = SymbolicAggregateApproximation(alphabet=self.alphabet, strategy=self.strategy_sax)
        self.swindow = None

    def term_frequency(self, word, count, n_words):
        self.tf[word] = count / n_words

    def inverse_data_frequency(self, N):
        for word, count in self.doc_contain_words.items():
            if count > 0:
                self.idf[word] = np.log(N / count)
            else:
                self.idf[word] = 0

    def get_sax_word(self, seq, window):
        X = self.paa.transform(seq, big_window=window)
        if X.size == 0:
            return [], ""
        self.sax.fit(seq.y)
        word_arr = self.sax.transform(X)
        return word_arr, "".join(word_arr)

    def bow(self, ts):
        self.swindow.initialize(ts)
        doc = []
        count_words = self.alphabet.get_count_words_dict()
        n_words = 0
        while True:
            seq = self.swindow.get_sequence()
            if seq.size() > 0:
                word_vec, word = self.get_sax_word(seq, self.swindow.window)
                if len(word_vec) == self.n:
                    doc.append(word)
                    count_words[word] += 1
                    n_words += 1
            if self.swindow.finish():
                break
            self.swindow.advance()
        return doc, count_words, n_words

    def set_sliding_window(self, swindow):
        self.swindow = swindow

    def dataset_to_bopf(self, D):
        tf_dataset = []
        D_new = []
        count_words_dataset = []
        for i, ts in enumerate(D):
            doc, count_words, n_words = self.bow(ts)
            count_words_dataset.append(count_words)
            if len(doc) > 0:
                tfDict = self.alphabet.get_count_words_dict()
                for word, count in count_words.items():
                    if count > 0:
                        self.doc_contain_words[word] += 1
                    tfDict[word] = count / n_words
                tf_dataset.append(tfDict)
                D_new.append(ts)

        self.inverse_data_frequency(len(D))
        vectors = []
        for i, tfDict in enumerate(tf_dataset):
            vec = self.alphabet.get_all_words_vec(self.n)
            for word, tf in tfDict.items():
                tf_idf = tf * self.idf[word]
                vec[self.alphabet.word_to_number(word)] = tf_idf
            vectors.append(vec)
        return vectors, D_new, count_words_dataset

