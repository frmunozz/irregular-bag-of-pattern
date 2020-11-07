from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import balanced_accuracy_score
import os


def _load_file(filename, **kwargs):
    has_time = kwargs.pop("has_time", False)
    sep = kwargs.get("sep", '\t')
    file1 = open(filename, 'r')
    lines = file1.readlines()
    m = len(lines)
    dataset = []
    times = []
    labels = []
    for d in lines:
        arr = d[:-1].split(sep)
        y = np.array(arr[1:], dtype=float)
        t = np.arange(y.size, dtype=float)
        dataset.append(y)
        times.append(t)
        labels.append(int(arr[0]))

    return dataset, times, labels, m


def _load_pandas(path,  **kwargs):
    data_filename = kwargs.pop("data_filename")
    meta_filename = kwargs.pop("meta_filename")
    df = pd.read_csv(path + data_filename)
    df_metadata = pd.read_csv(path + meta_filename)

    passband_id = 3

    df = df[df["passband"] == passband_id]
    df = df.sort_values(by=["object_id", "mjd"])
    df_metadata = df_metadata.sort_values(by=["object_id"])
    df = df.groupby("object_id")
    fluxes = df['flux'].apply(list)
    times = df['mjd'].apply(list)
    ids = df.groups.keys()
    dataset = [np.array(fluxes.loc[i]) for i in ids]
    times_arr = []
    for i in ids:
        times_i = np.array(times.loc[i])
        times_i = times_i - times_i[0]
        times_arr.append(times_i)

    labels = df_metadata["target"].to_numpy()

    return dataset, times_arr, labels, len(dataset)


def _load_numpy(data_path, **kwargs):
    n1 = kwargs.get("n1")
    n2 = kwargs.get("n2")
    c = kwargs.get("c")
    set_type = kwargs.get("set_type")
    dataset = np.load(data_path + "{}_d_n{}_c{}.npy".format(set_type, n1, c), allow_pickle=True)
    times = np.load(data_path + "{}_t_n{}_c{}.npy".format(set_type, n1, c), allow_pickle=True)
    labels = np.load(data_path + "{}_l_n{}_c{}.npy".format(set_type, n1, c), allow_pickle=True)
    return dataset, times, labels, len(dataset)


def load_numpy_dataset(data_path, file_base):
    dataset = np.load(os.path.join(data_path, file_base % "d"), allow_pickle=True)
    times = np.load(os.path.join(data_path, file_base % "t"), allow_pickle=True)
    labels = np.load(os.path.join(data_path, file_base % "l"), allow_pickle=True)
    return dataset, times, labels, len(dataset)


class BagOfPatternFeature(object):
    def __init__(self, special_character=True, strategy="special2"):
        self.dataset = []
        self.times = []
        self.labels = []
        self.m = 0
        self.cum1 = []
        self.cum2 = []
        self.bopsize = 0
        self.train_bop = None
        self.train_bop_matrix = None
        self.class_word_count = None
        self.train_label_index = None
        self.tlabel = None
        self.class_count = None
        self.c = 0
        self.bop_f_a = None
        self.sort_index = None
        self.fea_num = 0
        self.train_bop_sort = None
        self.crossL = None
        self.crossL2 = None
        self.best_idx = -1
        self.best_score = -1
        self.best2_idx = -1
        self.best2_score = -1
        self.class_positions = {}
        self.special_character = special_character
        self.dists_centroids = None
        self.dists_tf_idfs_p1 = None
        self.dists_tf_idfs_p2 = None
        self.dists_tf_idfs_p3 = None
        self.strategy = strategy
        self.alphabet_size = 4
        self.dropped_ts = []
        self.new_index = []
        self.new_m = -1

    def load_dataset(self, dataset_path, fmt="pandas", **kwargs):
        if fmt == "pandas":
            self.dataset, self.times, self.labels, self.m = _load_pandas(dataset_path, **kwargs)

        elif fmt == "file":
            self.dataset, self.times, self.labels, self.m = _load_file(dataset_path, **kwargs)
        elif fmt == "npy":
            # self.dataset, self.times, self.labels, self.m = _load_numpy(dataset_path, **kwargs)
            self.dataset, self.times, self.labels, self.m = load_numpy_dataset(dataset_path, kwargs.get("base", None))
        else:
            raise ValueError("fmt unknown")

    def cumsum(self):
        self.cum1 = []
        self.cum2 = []
        psum = np.zeros(self.m)
        psum2 = np.zeros(self.m)
        for i in range(self.m):
            data = self.dataset[i]
            n = data.size
            cum1_data = np.zeros(n+1)
            cum2_data = np.zeros(n+1)
            for j in range(n):
                psum[i] += data[j]
                psum2[i] += data[j] ** 2
                cum1_data[j+1] = psum[i]
                cum2_data[j+1] = psum2[i]
            self.cum1.append(cum1_data)
            self.cum2.append(cum2_data)

    def _bop_get_next_sequence(self, time_stamps, n, i, j, window, tol=15):
        while j < n:
            if time_stamps[j] - time_stamps[i] < window:
                j += 1
            else:
                if j - i - 1 > tol:
                    seq_ini = i
                    seq_end = j
                    i += 1
                    return seq_ini, seq_end, i, j
                else:
                    i += 1
                    j += 1

        # if j == n:
        #     return -2, -2, i, j
        if j - i - 1 < tol:
            return -1, -1, i, j
        else:
            return i, j, i, j

    def _bop_get_next_segment(self, time_stamps, n, w_i, seq_i,  seg_j, seg_window, k):
        # if k == 34 and seq_i == 270:
        #     print("before: ", time_stamps[seg_j], time_stamps[seq_i], seg_window * (w_i + 1))
        ini_time = time_stamps[seq_i]
        cmp = seg_window * (w_i + 1) + ini_time
        # while time_stamps[seg_j] < cmp:
        #     if seg_j < n - 1 and time_stamps[seg_j + 1] < cmp:
        #         seg_j += 1
        #     else:
        #         break

        while seg_j < n and time_stamps[seg_j] < cmp:
            if seg_j == n:
                break
            seg_j += 1
        # if k == 34 and seq_i == 270:
        #     print("after: ", time_stamps[seg_j], time_stamps[seq_i], seg_window * (w_i + 1))
        return seg_j

    def bop_paper(self, wd, wl):
        self.bopsize = 1 << (2 * wd)
        self.train_bop = np.zeros((self.m * self.bopsize) + 1)
        self.train_bop_matrix = np.zeros((self.m, self.bopsize))
        for i in range(self.m):
            pword = -1
            data = self.dataset[i]
            n = data.size
            wl_n = int(round(wl * n))
            ns = wl_n / wd
            print(wl_n, ns)
            cum1_data = self.cum1[i]
            cum2_data = self.cum2[i]
            for j in range(n - wl_n + 1):
                sumx = cum1_data[j + wl_n] - cum1_data[j]
                sumx2 = cum2_data[j + wl_n] - cum2_data[j]
                meanx = sumx / wl_n
                sigmax = np.sqrt(sumx2 / wl_n - meanx*meanx)
                wordp = 0
                for k in range(wd):
                    u = int(round(ns * (k + 1)))
                    l = int(round(ns * k))
                    sumsub = cum1_data[j + u] - cum1_data[j + l]
                    avgsub = sumsub / (u - l)
                    paa = (avgsub - meanx) / sigmax
                    if paa < 0:
                        if paa < -0.67:
                            val = 0
                        else:
                            val = 1
                    else:
                        if paa < 0.67:
                            val = 2
                        else:
                            val = 3
                    ws = (1 << (2 * k))*val
                    wordp += ws
                if pword != wordp:
                    self.train_bop[i + wordp * self.m] += 1
                    self.train_bop_matrix[i][wordp] += 1
                    pword = wordp

    def set_bopsize(self, wd):
        if self.special_character:
            self.bopsize = 1 << (2 * (wd + 1))
        else:
            self.bopsize = 1 << (2 * wd)

    def set_word_feature(self, pword, wordp, k):
        if pword != wordp:
            self.train_bop[k + wordp * self.m] += 1
            self.train_bop_matrix[k][wordp] += 1
            pword = wordp
        return pword

    def word_feature_cases(self, pword, k, checker_empty, segment_characters):
        s = np.sum(checker_empty)
        if self.special_character:
            if self.strategy == "special1":
                # special1: allow a 'threshold' number of 'special character' on each word
                threshold = len(checker_empty) // 2.3
                # threshold is defined in way that: for wd = 3, 4, threshold = 1, for wd = 5, 6, threshold = 2, for wd=7, threshold = 3
                wordp = int(np.sum(segment_characters))
                if s > threshold:
                    return pword
                else:
                    return self.set_word_feature(pword, wordp, k)
            elif self.strategy == "special2":
                # special2: allow at most 2 'special character' at beginning and/or end of the word
                # on wd > 5 allows 2 'special characters', and on wd < 5, allows only 1
                wordp = int(np.sum(segment_characters))
                wd = len(checker_empty)
                if s == 2 and wd > 5:
                    if checker_empty[0] == 1 and checker_empty[-1] == 1:
                        pword = self.set_word_feature(pword, wordp, k)
                elif s == 1:
                    if checker_empty[0] == 1 or checker_empty[-1] == 1:
                        pword = self.set_word_feature(pword, wordp, k)
                elif s == 0:
                    pword = self.set_word_feature(pword, wordp, k)
                return pword

            elif self.strategy == "special3":
                # on "special3" we will take "special character" as a joker and add them to different possibilities
                raise NotImplementedError()
            else:
                raise ValueError("unknown strategy '%s'" % self.strategy)



        else:
            if s > 0:
                return pword
            else:
                # we are on regular case
                wordp = int(np.sum(segment_characters))
                return self.set_word_feature(pword, wordp, k)

    def bop(self, wd, wl, tol=1, verbose=True, window_type="fraction"):
        self.set_bopsize(wd)
        self.train_bop = np.zeros((self.m * self.bopsize) + 1)
        self.dropped_ts = []
        self.new_index = []
        self.train_bop_matrix = np.zeros((self.m, self.bopsize))
        count_empty_segments = 0
        count_small_ts = 0
        for k in range(self.m):
            pword = -1
            data = self.dataset[k]
            time_stamps = self.times[k]
            n = data.size
            i = 0
            j = 1
            cum1_data = self.cum1[k]
            cum2_data = self.cum2[k]
            ts_width = time_stamps[n-1] - time_stamps[0]
            if n < 6:
                count_small_ts += 1
            if window_type == "fraction":
                window = (1.0 * wl) * ts_width
            else:
                window = wl
            accepted = False
            while j < n and window <= ts_width:
                seq_i, seq_j, i, j = self._bop_get_next_sequence(time_stamps, n, i, j, window, tol=tol)
                if seq_i == seq_j and seq_i == -1:
                    break
                sumx = cum1_data[seq_j] - cum1_data[seq_i]
                sumx2 = cum2_data[seq_j] - cum2_data[seq_i]
                meanx = sumx / (seq_j - seq_i)
                # if round(sumx2 / (seq_j-seq_i), 5) < round(meanx * meanx, 5):
                #     print(k, i, j, seq_i, seq_j, meanx, meanx*meanx, sumx2, sumx2 / (seq_j - seq_i))
                sigmax = np.sqrt(round(sumx2 / (seq_j-seq_i), 15) - round(meanx * meanx, 15))
                seg_j = seq_i
                seg_window = window / wd
                checker_empty = np.zeros(wd)
                segment_characters = np.zeros(wd)
                for w_i in range(wd):
                    seg_i = seg_j
                    seg_j = self._bop_get_next_segment(time_stamps, n, w_i, seq_i, seg_j, seg_window, k)
                    if seg_j - seg_i <= tol // wd:
                        count_empty_segments += 1
                        checker_empty[w_i] = 1
                        val = 4

                    else:
                        sumsub = cum1_data[seg_j] - cum1_data[seg_i]
                        avgsub = sumsub / (seg_j - seg_i)
                        if False: #sigmax == 0:
                            paa = 0
                        else:
                            paa = (avgsub - meanx) / sigmax
                        # if k == 0 and seq_i >= 0:
                        #     print("i: {}, j: {}, seg_i: {}, seg_j: {}, seg_win: {}, offset: {}, paa: {}".format(seq_i, seq_j, seg_i,
                        #                                                                            seg_j, seg_window, seg_window * (w_i + 1), paa))
                        if paa < 0:
                            if paa < -0.67:
                                val = 0
                            else:
                                val = 1
                        else:
                            if paa < 0.67:
                                val = 2
                            else:
                                val = 3
                    ws = (1 << (2 * w_i))*val
                    segment_characters[w_i] = ws
                pword = self.word_feature_cases(pword, k, checker_empty, segment_characters)
                if pword != -1:
                    accepted = True
            if not accepted:
                self.dropped_ts.append(k)
                self.labels[k] = 111
            else:
                self.new_index.append(k)
        self.new_m = len(self.new_index)
        if verbose:
            print("TOTAL DE SEGMENTOS VACIOS: ", count_empty_segments)
            print("TOTAL DE TIMESERIES DESCARTADAS: ", len(self.dropped_ts))
            print("TOTAL DE ts < 6:" , count_small_ts)

    def adjust_label_set(self):
        self.train_label_index = np.zeros(self.m, dtype=int)
        self.tlabel = np.sort(np.unique(self.labels))[:-1]
        self.c = self.tlabel.shape[0]
        self.class_count = np.zeros(self.c)
        self.class_positions = defaultdict(list)
        for i in range(self.m):
            l = self.labels[i]
            if l != 111:
                position = int(np.where(self.tlabel == l)[0][0])
                self.train_label_index[i] = position
                self.class_positions[self.tlabel[position]].append(i)
                self.class_count[position] += 1

    def anova_matrix(self, verbose=True):
        self.bop_f_a = np.zeros(self.bopsize)
        f_inputs = []
        for i in range(self.c):
            idxs = np.where(i == self.train_label_index)[0]
            f_inputs.append(self.train_bop_matrix[idxs])
        f, p = f_oneway(*f_inputs)
        self.bop_f_a = np.round(np.nan_to_num(f), 5)

    def anova(self, verbose=True):
        self.bop_f_a = np.zeros(self.bopsize)
        bop_f_a_i = 0
        for j in range(self.bopsize):
            if j % 100 == 0 and verbose:
                print("{}/{}".format(j, self.bopsize), end="\r")
            sumall = 0.0
            sumc = np.zeros(self.c)
            for i1 in range(self.m):
            # for i1 in self.new_index:
                k = int(self.train_label_index[i1])
                sumall += self.train_bop[i1 + j * self.m]
                sumc[k] += self.train_bop[i1 + j * self.m]

            avgall = sumall / self.m
            # avgall = sumall / self.new_m
            ssa = 0.0
            avgc = np.zeros(self.c)
            for k in range(self.c):
                avgc[k] = sumc[k] / self.class_count[k]
                ssa += self.class_count[k] * (avgc[k] - avgall) * (avgc[k] - avgall)

            ssw = 0.0
            for i2 in range(self.m):
            # for i2 in self.new_index:
                k = int(self.train_label_index[i2])
                ssw += (self.train_bop[i2 + j * self.m] - avgc[k]) * (self.train_bop[i2 + j * self.m] - avgc[k])

            msa = 1.0 * ssa / (self.c - 1)
            msw = 1.0 * ssw / (self.m - self.c)
            # msw = 1.0 * ssw / (self.new_m - self.c)
            if msa == 0 and msw == 0:
                self.bop_f_a[bop_f_a_i] = 0
            elif msw == 0 and msa != 0:
                self.bop_f_a[bop_f_a_i] = np.inf
            else:
                self.bop_f_a[bop_f_a_i] = np.round(msa / msw, 5)

            # if msa != 0 or msw != 0:
            #     print("j={}, msw={}, msw={}".format(j, msa, msw))
            bop_f_a_i += 1

    def anova_sort(self):
        self.sort_index = np.argsort(self.bop_f_a)[::-1]

        self.fea_num = 0
        while self.fea_num < self.bopsize and self.bop_f_a[self.sort_index[self.fea_num]] > 0:
            self.fea_num += 1

    def sort_trim_arr(self, verbose=True):
        train_bop_sort = []
        for j in range(self.fea_num):
            if j % 100 == 0 and verbose:
                print("{}/{}".format(j, self.fea_num), end="\r")
            k = self.sort_index[j]
            idxs = np.arange(self.m) + k * self.m
            train_bop_sort.extend(self.train_bop[idxs])
            # for i in range(self.m):
            #     train_bop_sort.append(self.train_bop[i + k * self.m])
        self.train_bop_sort = np.array(train_bop_sort)

    def count_words_by_class(self):
        self.class_word_count = np.zeros((self.c, self.fea_num))
        for k in range(self.c):
            idxs = np.where(k == self.train_label_index)[0]
            self.class_word_count[k] += np.sum(self.train_bop_sort[idxs], axis=0)

    def get_cv_centroid(self, k, j):
        self.class_word_count[k] -= self.train_bop_sort[j]
        centroid = self.get_centroid()
        self.class_word_count[k] += self.train_bop_sort[j]
        return centroid

    def get_centroid(self):
        centroids = np.zeros((self.c, self.fea_num))

        for k in range(self.c):
            centroids[k] = np.divide(self.class_word_count[k], self.class_count[k])

        return centroids

    def centroid_eval(self, centroids, i, j):
        rmin = np.inf
        rmin_idx = -1
        for k in range(self.c):
            self.dists_centroids[k] += (self.train_bop_sort[j + i * self.m] - centroids[k][i]) ** 2
            if rmin > self.dists_centroids[k]:
                rmin = self.dists_centroids[k]
                rmin_idx = k
        return rmin, rmin_idx

    def cross_VL(self):
        # x = np.zeros((self.c, self.fea_num))
        # y = np.zeros((self.m, self.c))
        self.crossL = []
        maxcount = -1
        maxk = -1
        self.dists_centroids = np.zeros(self.c)

        for i in range(self.fea_num):
            cv_centroid_matchs = 0
            for j in range(self.m):
                k = self.train_label_index[j]
                centroids = self.get_cv_centroid(k, j)
                # rmin_c, rmin_c_idx = self.centroid_eval(centroids, i, j)

                # if rmin_c_idx == k:
                #     cv_centroid_matchs += 1

            # if cv_centroid_matchs > maxcount:
            #     maxcount = cv_centroid_matchs
            #     maxk = i

        # self.best_idx = maxk + 1
        # self.best_score = maxcount / self.m

        # self.crossL = self.get_centroid()

    def crossVL(self, verbose=True):
        x = np.zeros((self.c, self.fea_num))
        y = np.zeros((self.m, self.c))
        self.crossL = []
        maxcount = -1
        maxk = -1
        maxacc = -np.inf
        maxk2 = -1
        for k in range(self.fea_num):
            if verbose:
                print("{}/{}".format(k, self.fea_num), end="\r")
            count = 0
            for i in range(self.m):
            # for i in self.new_index:
                p = self.train_label_index[i]
                # if k == 0:
                    # print("train_bop_sort k={}, i={} :".format(k, i), self.train_bop_sort[i + k*self.m])
                x[p][k] += self.train_bop_sort[i + k * self.m]

            # print("x for k =", k, " :", x[:, k])

            for i in range(self.c):
                x[i][k] = x[i][k] / self.class_count[i]
                self.crossL.append(x[i][k])

            real_labels = []
            pred_labels = []
            for i in range(self.m):
            # for i in self.new_index:
                rmin = np.inf
                label = 0.0
                p = self.train_label_index[i]
                countc = self.class_count[p]
                for j in range(self.c):
                    r = y[i][j]
                    pm = self.train_bop_sort[i + k * self.m]
                    d = pm - x[j][k]
                    if j == p:
                        d += pm / countc
                    r += d * d
                    y[i][j] = r
                    if r < rmin:
                        rmin = r
                        label = j
                real_labels.append(self.train_label_index[i])
                pred_labels.append(label)
                if label == self.train_label_index[i]:
                    count += 1
            # print(k+1, count / self.m)
            balanced_acc = balanced_accuracy_score(real_labels, pred_labels)
            if count >= maxcount:
                maxcount = count
                maxk = k

            if balanced_acc >= maxacc:
                maxacc = balanced_acc
                maxk2 = k

        # print("crossVL using imbalance acc: ", maxk + 1, ", crossVL using balanced acc:", maxk2+1)
        # self.best_idx = maxk + 1
        # self.best_score = maxcount / self.m

        self.best_idx = maxk2 + 1
        self.best_score = maxacc

    def crossVL2(self):
        x = np.zeros((self.c, self.fea_num))
        y1 = np.zeros((self.m, self.c))
        y2 = np.zeros((self.m, self.c))
        y3 = np.zeros((self.m, self.c))

        self.crossL2 = []
        maxcount = -1
        maxacc = -np.inf
        maxk = -1
        maxk2 = -1
        for k in range(self.fea_num):
            count = 0
            for i in range(self.m):
            # for i in self.new_index:
                p = self.train_label_index[i]
                x[p][k] += self.train_bop_sort[i + k*self.m]
            countc = 0.0
            for i in range(self.c):
                if x[i][k] > 0:
                    countc += 1

            for i in range(self.c):
                if x[i][k] > 0:
                    x[i][k] = (1 + np.log10(x[i][k]))*(np.log10(1 + self.c/countc))
                self.crossL2.append(x[i][k])

            real_labels = []
            pred_labels = []
            for i in range(self.m):
            # for i in self.new_index:
                rmax = -np.inf
                label = 0.0
                for j in range(self.c):
                    r1 = y1[i][j]
                    r2 = y2[i][j]
                    r3 = y3[i][j]
                    d = self.train_bop_sort[i + k*self.m]
                    if d > 0:
                        d = 1 + np.log10(d)
                    r1 += d * x[j][k]
                    r2 += d*d
                    r3 += x[j][k] * x[j][k]
                    y1[i][j] = r1
                    y2[i][j] = r2
                    y3[i][j] = r3
                    if r2 != 0 and r3 != 0:
                        r = r1*r1 / (r2*r3)
                        if r > rmax:
                            rmax = r
                            label = j
                real_labels.append(self.train_label_index[i])
                pred_labels.append(label)
                if label == self.train_label_index[i]:
                    count += 1
                
            balanced_acc = balanced_accuracy_score(real_labels, pred_labels)

            if count >= maxcount:
                maxcount = count
                maxk = k

            if balanced_acc >= maxacc:
                maxacc = balanced_acc
                maxk2 = k

        # print("crossVL2 using imbalance acc: ", maxk + 1, ", crossVL using balanced acc:", maxk2 + 1)
        # self.best2_idx = maxk+1
        # self.best2_score = maxcount / self.m

        self.best2_idx = maxk2 + 1
        self.best2_score = maxacc

        return self.crossL2

    def classify(self, test_bop, mt):
        train_bop_centroids = self.crossL
        res = np.zeros(mt, dtype=int)
        for i in range(mt):
            rmin = np.inf
            label = -1
            for j in range(self.c):
                r = 0.0
                pm = test_bop[i]
                pv = train_bop_centroids[j]
                for k in range(self.best_idx):
                    d = pm - pv
                    pm = test_bop[i + (k+1)*mt]
                    pv = train_bop_centroids[i + (k+1)*self.c]
                    r += d*d
                if r < rmin:
                    rmin = r
                    label = self.tlabel[j]
            res[i] = label

        return res
