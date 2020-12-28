import numpy as np
from collections import defaultdict


class Alphabet(object):
    def __init__(self, letters):
        '''Alphabet

        a class to define some methods for the alphabet used.
        '''
        if isinstance(letters, list):
            letters = np.array(letters)
        self.letters = letters
        self.letter_numbers = {l: i for i, l in enumerate(self.letters)}

    def size(self):
        return self.letters.size

    def word_to_number(self, word):
        l = len(word) - 1
        i = l
        num = 0
        while i >= 0:
            j = l- i
            if j == 0:
                num += self.letter_numbers[word[i]]
            else:
                num += self.letter_numbers[word[i]] * self.size() ** j
            i -= 1
        return num

    def get_all_words_vec(self, n):
        return np.zeros(self.size() ** n)

    def get_word(self, idxs):
        return self.letters[idxs]

    def get_count_words_dict(self):
        count_bow = defaultdict(int)
        n = len(self.letters)
        for j in range(n):
            for k in range(n):
                count_bow[self.letters[j] + self.letters[k]] = 0
        return count_bow