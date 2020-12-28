from collections import defaultdict
from .bop import features_sax_size
import numpy as np
from scipy.spatial import distance


class ANOVACVFeatures(object):
    def __init__(self, train_labels, alphabet_size, train_bop):
        self.train_labels = train_labels
        self.train_class_count = self.count_labels()
        self.labels_index, self.index_label = self.get_labels_index()
        self.alphabet_size = alphabet_size
        self.train_bop = train_bop
        self.class_count_words = self.get_count_words_by_class(train_bop)
        n_classes = self.class_count_words.shape[0]
        self.dists_centroids = np.zeros(n_classes)
        self.dists_tf_idfs_p1 = np.zeros(n_classes)
        self.dists_tf_idfs_p2 = np.zeros(n_classes)
        self.dists_tf_idfs_p3 = np.zeros(n_classes)

    def count_labels(self):
        train_class_count = defaultdict(int)
        for l in self.train_labels:
            train_class_count[l] += 1
        return train_class_count

    def get_labels_index(self):
        label_idx = {}
        idx_label = {}
        for i, label in enumerate(self.train_class_count):
            label_idx[label] = i
            idx_label[i] = label
        return label_idx, idx_label

    def get_count_words_by_class(self, train_bop):
        class_count_words = np.zeros((len(self.train_class_count), features_sax_size(self.alphabet_size)), dtype=float)
        for label, word_count in zip(self.train_labels, train_bop):
            class_count_words[self.labels_index[label]] += word_count
        return class_count_words

    def get_class_count_by_word(self):
        words_count_class = np.zeros(features_sax_size(self.alphabet_size), dtype=int)
        for class_counts in self.class_count_words:
            for i, count in enumerate(class_counts):
                if count > 0:
                    words_count_class[i] += 1
        return words_count_class

    def get_centroids(self):
        centroids = np.zeros(self.class_count_words.shape)

        # divide each sum by the count of time series on each class
        for label, i in self.labels_index.items():
            centroids[i] = np.divide(self.class_count_words[i], self.train_class_count[label])
        return centroids

    def get_cv_centroids(self, i, train_bop_count):
        self.class_count_words[i] -= train_bop_count
        centroid = self.get_centroids()
        self.class_count_words[i] += train_bop_count
        return centroid

    def get_tf_idf(self, words_count_class):
        tf_idf = np.zeros(self.class_count_words.shape)
        c = self.class_count_words.shape[0]
        for label, i in self.labels_index.items():
            for j, count in enumerate(self.class_count_words[i]):
                if count > 0:
                    tf_idf[i][j] = (1 + np.log(count)) * np.log(1 + (c / words_count_class[j]))

        return tf_idf

    def get_cv_tf_idf(self, i, train_bop_count):
        self.class_count_words[i] -= train_bop_count
        words_count_class = self.get_class_count_by_word()
        tf_idf = self.get_tf_idf(words_count_class)
        self.class_count_words[i] += train_bop_count
        return tf_idf

    def set_sum_vectors(self):
        self.dists_centroids[:] = 0
        self.dists_tf_idfs_p1[:] = 0
        self.dists_tf_idfs_p2[:] = 0
        self.dists_tf_idfs_p3[:] = 0

    def _cv_centroid_test(self, centroids, ts_word_counts, next_idx):
        rmin = np.inf
        rmin_idx = np.inf
        for j, centroid_j in enumerate(centroids):
            self.dists_centroids[j] += (ts_word_counts[next_idx] - centroid_j[next_idx]) ** 2
            if rmin > self.dists_centroids[j]:
                rmin = self.dists_centroids[j]
                rmin_idx = j
        return rmin, rmin_idx

    def _cv_tf_idf_test(self, tf_idfs, ts_word_counts, next_idx):
        rmin = np.inf
        rmin_idx = np.inf
        for j, tf_idf_j in enumerate(tf_idfs):
            self.dists_tf_idfs_p1[j] += ts_word_counts[next_idx] * tf_idf_j[next_idx]
            self.dists_tf_idfs_p2[j] += ts_word_counts[next_idx]
            self.dists_tf_idfs_p3[j] += tf_idf_j[next_idx]
            square_cosine_similarity = (self.dists_tf_idfs_p1[j] ** 2) / (self.dists_tf_idfs_p2[j] * self.dists_tf_idfs_p3[j])
            if rmin > square_cosine_similarity:
                rmin = square_cosine_similarity
                rmin_idx = j
        return rmin, rmin_idx

    def cv_run(self, sorted_anova_rank):
        self.set_sum_vectors()
        anova_c_max = -1
        anova_c_max_idx = 0
        anova_tf_idf_max = -1
        anova_tf_idf_max_idx = 0
        for i in range(len(sorted_anova_rank)):
            next_idx = sorted_anova_rank[i]
            cv_centroid_match = 0
            cv_tf_idf_match = 0
            for label, word_count in zip(self.train_labels, self.train_bop):
                idx = self.labels_index[label]
                centroids = self.get_cv_centroids(idx, word_count)
                tf_idfs = self.get_cv_tf_idf(idx, word_count)

                rmin_c, rmin_c_idx = self._cv_centroid_test(centroids, word_count, next_idx)
                rmin_tf_idf, rmin_tf_idf_idx = self._cv_tf_idf_test(tf_idfs, word_count, next_idx)
                expected_rmin_idx = self.labels_index[label]

                if rmin_c_idx == expected_rmin_idx:
                    cv_centroid_match += 1
                if rmin_tf_idf_idx == expected_rmin_idx:
                    cv_tf_idf_match += 1

            cv_centroid_match /= float(len(self.train_labels))
            cv_tf_idf_match /= float(len(self.train_labels))

            print("centroid match for best {} anova F: {}".format(i, cv_centroid_match))
            print("tf_idf match for best {} anovaF: {}".format(i, cv_tf_idf_match))
            if anova_c_max <= cv_centroid_match:
                anova_c_max = cv_centroid_match
                anova_c_max_idx = i

            if anova_tf_idf_max <= cv_tf_idf_match:
                anova_tf_idf_max = cv_tf_idf_match
                anova_tf_idf_max_idx = i

        return anova_c_max_idx, anova_tf_idf_max_idx
