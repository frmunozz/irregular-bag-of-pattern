from .alphabet import Alphabet
import numpy as np
from scipy.stats import norm


class SymbolicAggregateApproximation(object):
    def __init__(self, alphabet=None, strategy="normal", special_character=False):
        if alphabet is None:
            alphabet = Alphabet(['a', 'b', 'c', 'd'])
        elif isinstance(alphabet, list):
            alphabet = Alphabet(alphabet)

        if not strategy == 'normal':
            raise ValueError("Strategy '{}' not implemented".format(strategy))

        self.alphabet = alphabet
        self.strategy = strategy
        self.break_points = None
        self.special_character = special_character

    def fit(self, X):
        """
        fit the break points according to the selected strategy

        :param X: the values used for the fit
        :return: an array with the bins
        """

        std = np.std(X)
        mean = np.mean(X)
        self.break_points = norm.ppf(np.linspace(0, 1, self.alphabet.size() + 1)[1:-1], loc=mean, scale=std)

    def transform(self, uts):
        """
        transform the corresponding UTS to a word
        :param uts: time series
        :return: a vector with the letter of the word
        """
        if self.special_character:
            self.break_points = np.append(self.break_points, np.inf)

        discretized = np.digitize(uts.y, self.break_points)
        return self.alphabet.get_word(discretized)
