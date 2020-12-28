from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from src.utils import AbstractCore
from .transformer import CountVectorizer, transformer_mp
from .classify import cv_classify, simple_train_test_classify
from .class_vectors import predict_by_centroid, predict_by_tf_idf, compute_class_centroids, compute_class_tf_idf

import multiprocessing as mp
import queue
import numpy as np
from scipy import sparse
import copy
from collections import defaultdict
import time


_VALID_KWARGS = {
    "early_stopping": 5,
    "threshold1": 50000,
    "threshold2": 100000,
    "best_n_lengths": 4,
    "best_n_windows": 10,
    "max_n_windows": 30,
    "max_n_lengths": 7,
    "verbose": True,
    "multiprocessing": True,
}


def merge_count_vectors(count_vectors, by="window"):
    if by == "window":
        pass
    elif by == "word_length":
        pass


class CVParamFinder(AbstractCore):

    def __init__(self, **kwargs):
        """
        use Cross-Validation to find which are the best param combination.

        To reduce the grades of freedom in combinations to try, the following procedure is adopted:

            1. for
                    window = mean_length/2
                    word_length = [1, 2, 3, ...] (stop when feature size is higher than threshold1)
                find best 4 word_length

            2. for
                    word_length = best word_length of 1.
                    window=linspace(word_length, mean_width, min(30, mean_width))
                find best min(10, mean_width // 3) windows

            3. for each possible combination given by 1. and 2., compute incrementally
                the concatenated vectors and classify by kfold.

            4. stop when the number of features reach a threshold2 or by early stopping criteria.


        for each cross-validation StratifiedKFold is used, with n_splits to be defined

        :param kwargs:
        """
        super(CVParamFinder, self).__init__(**kwargs)
        self.selected_tuples=[]
        self.selected_vectors = None
        self.selected_wls_for_cv = []
        self.selected_wins_for_cv = []
        self.params = defaultdict(dict)
        self.all_vectors = None
        self.wl_arr = []
        self.win_arr = []
        self.global_best_bacc = -1
        self.stop = False

    def get_valid_kwargs(self) -> dict:
        return copy.deepcopy(_VALID_KWARGS)

    @classmethod
    def module_name(cls):
        return "MethodA"

    def get_concatenated_vectors(self, vectors):
        if self.selected_vectors is None:
            return vectors
        else:
            return sparse.hstack((self.selected_vectors, vectors)).tolil()

    def _worker_bacc(self, labels, vectors_arr, wl_arr, win_arr, lock, comb_to_try, r_queue, n, **kwargs):
        while True:
            try:
                lock.acquire()
                i = comb_to_try.get_nowait()
                vectors = vectors_arr[i]
                wl = wl_arr[i]
                win = win_arr[i]

            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                if (win, wl) not in self.selected_tuples:
                    real, pred = cv_classify(vectors, labels, use_pca=kwargs.get("use_pca", True),
                                         cv_method=kwargs.get("cv_method", "loo"),
                                         class_method=kwargs.get("class_method", "centroid"),
                                         n_splits=kwargs.get("n_splits", 5),
                                         scale=kwargs.get("scale", True),
                                             with_mean=kwargs.get("with_mean", True),
                                             n_components=kwargs.get("n_components", 20),
                                             dist_method=kwargs.get("dist_method", "cosine"),
                                             repr_method=kwargs.get("repr_method", "bopf"))
                    real_labels = []
                    pred_labels = []
                    for real_i, pred_i in zip(real, pred):
                        real_labels.extend(list(real_i))
                        pred_labels.extend(list(pred_i))

                    bacc = balanced_accuracy_score(real_labels, pred_labels)
                    r_queue.put((i, bacc))
                    print("|", end="")
                else:
                    r_queue.put((i, -1))

    def multiprocessing_bacc_arr(self, vectors_arr, wl_arr, win_arr, n, labels, n_process="default", **kwargs):
        if n_process == "default":
            n_process = mp.cpu_count()
            # n_process = 1

        print("computing balanced accuracy for Cross-validation using %d process" % n_process)
        print("Multiprocessing CV: ", end="")
        m = mp.Manager()
        result_queue = m.Queue()

        combinations_to_try = mp.Queue()

        for i in range(n):
            combinations_to_try.put(i)

        lock = mp.Lock()
        # self._worker_bacc(labels, vectors_arr, wl_arr, win_arr, lock, combinations_to_try, result_queue, n, **kwargs)
        jobs = []
        for w in range(n_process):
            p = mp.Process(target=self._worker_bacc,
                           args=(labels, vectors_arr, wl_arr, win_arr, lock, combinations_to_try, result_queue, n),
                           kwargs=kwargs)
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        print(" [DONE]")
        bacc_arr = [None] * n
        num_res = result_queue.qsize()
        while num_res > 0:
            i, bacc = result_queue.get()
            bacc_arr[i] = bacc
            num_res -= 1

        return bacc_arr

    def cv_first_step(self, dataset, times, labels, **kwargs):
        mean_width = np.mean([x[-1] - x[0] for x in times])
        window = mean_width / 2
        word_length_max = int(np.log10(self["threshold1"])/np.log10(kwargs.get("alph_size", 4)))
        word_length = np.arange(min(self["max_n_lengths"], word_length_max)) + 1
        use_pca = kwargs.pop("use_pca", True)
        cv_method = kwargs.pop("cv_method", "loo")
        class_method = kwargs.pop("class_method", "centroid")
        n_splits = kwargs.pop("n_splits", 5)
        scale = kwargs.pop("scale", True)
        with_mean = kwargs.pop("with_mean", True)
        n_components = kwargs.pop("n_components", 20)
        dist_method = kwargs.pop("dist_method", "cosine")
        repr_method = kwargs.pop("repr_method", "bopf")

        # kwargs["window"] = window
        # kwargs["word_length"] = word_length
        kwargs["verbose"] = True
        transf = CountVectorizer(**kwargs)
        transf.fit(dataset, times)
        if self["verbose"]:
            print("computing count vectorizers for window = [{}], word_length = {}".format(window, word_length), end=" ")
        # wl_arr, win_arr, _, count_vec_arr = transf.transform(dataset, times)
        wl_arr, win_arr, count_vec_arr = transformer_mp(dataset, times, word_length, [window], **kwargs)
        if self["verbose"]:
            print("DONE")
        win = win_arr[0]
        n = len(wl_arr)
        # bacc_arr = []
        ini = time.time()
        bacc_arr = self.multiprocessing_bacc_arr(count_vec_arr, wl_arr, win_arr, n, np.array(labels),
                                                 use_pca=use_pca, cv_method=cv_method,
                                                 class_method=class_method, n_splits=n_splits,
                                                 scale=scale, with_mean=with_mean, n_components=n_components,
                                                 dist_method=dist_method, repr_method=repr_method)
        # for i in range(n):
        #     wl = wl_arr[i]
        #     vectors = count_vec_arr[i]
        #     real, pred = cv_classify(vectors, np.array(labels),use_pca=use_pca, cv_method=cv_method,
        #                              class_method=class_method, n_splits=n_splits)
        #     bacc = balanced_accuracy_score(real, pred)
        #     if self["verbose"]:
        #         print("first cv -> pair [%.1f, %d] (%d/%d) ==> balanced_acc:" % (win, wl, i+1, n),
        #               round(bacc, 3), end="\r")
        #     bacc_arr.append(bacc)
        end = time.time()

        sorted_best = np.argsort(bacc_arr)[::-1]
        sorted_best = sorted_best[:self["best_n_lengths"]]
        best_bacc = bacc_arr[sorted_best[0]]
        best_wl = wl_arr[sorted_best[0]]
        n_features = count_vec_arr[sorted_best[0]].shape[1]
        self.selected_wls_for_cv = np.array(wl_arr)[sorted_best]
        print("first cv -> best b_acc: %.3f for pair [%.1f, %d] (%d features) (time: %.3f sec)" % (round(best_bacc, 3),
                                                                                                   win, best_wl,
                                                                                                   n_features, end-ini))

    def cv_second_step(self, dataset, times, labels, **kwargs):
        best_wl = int(self.selected_wls_for_cv[0])
        mean_width = np.mean([x[-1] - x[0] for x in times])
        prev_window = mean_width / 2
        max_window_width = min(self["max_n_windows"], int(mean_width))
        windows = np.linspace(best_wl * 2, mean_width, max_window_width)
        windows = np.round(np.sort(np.append(windows, prev_window)), 2)
        use_pca = kwargs.pop("use_pca", True)
        cv_method = kwargs.pop("cv_method", "loo")
        class_method = kwargs.pop("class_method", "centroid")
        n_splits = kwargs.pop("n_splits", 5)
        scale = kwargs.pop("scale", True)
        with_mean = kwargs.pop("with_mean", True)
        n_components = kwargs.pop("n_components", 20)
        dist_method = kwargs.pop("dist_method", "cosine")
        repr_method = kwargs.pop("repr_method", "bopf")

        # kwargs["window"] = windows
        # kwargs["word_length"] = best_wl
        kwargs["verbose"] = True
        transf = CountVectorizer(**kwargs)
        transf.fit(dataset, times)
        if self["verbose"]:
            print("computing count vectorizers for window = {}, word_length = [{}]".format(windows, best_wl), end=" ")
        # _, win_arr, _, count_vec_arr = transf.transform(dataset, times)
        wl_arr, win_arr, count_vec_arr = transformer_mp(dataset, times, [best_wl], windows, **kwargs)
        if self["verbose"]:
            print("DONE")
        n = len(win_arr)
        # bacc_arr = []
        ini = time.time()
        bacc_arr = self.multiprocessing_bacc_arr(count_vec_arr, wl_arr, win_arr, n, np.array(labels),
                                                 use_pca=use_pca, cv_method=cv_method,
                                                 class_method=class_method, n_splits=n_splits,
                                                 scale=scale, with_mean=with_mean, n_components=n_components,
                                                 dist_method=dist_method, repr_method=repr_method)
        # for i in range(n):
        #     win = win_arr[i]
        #     vectors = count_vec_arr[i]
        #     real, pred = cv_classify(vectors, np.array(labels), use_pca=use_pca, cv_method=cv_method,
        #                              class_method=class_method, n_splits=n_splits)
        #     bacc = balanced_accuracy_score(real, pred)
        #     if self["verbose"]:
        #         print("first cv -> pair [%.1f, %d] (%d/%d) ==> balanced_acc:" % (win, best_wl, i+1, n),
        #               round(bacc, 3), end="\r")
        #     bacc_arr.append(bacc)
        end = time.time()
        sorted_best = np.argsort(bacc_arr)[::-1]
        sorted_best = sorted_best[:self["best_n_windows"]]
        best_bacc = bacc_arr[sorted_best[0]]
        best_win = win_arr[sorted_best[0]]
        n_features = count_vec_arr[sorted_best[0]].shape[1]
        self.selected_wins_for_cv = np.array(win_arr)[sorted_best]
        self.selected_tuples.append((best_win, best_wl))
        self.selected_vectors = count_vec_arr[sorted_best[0]]
        self.global_best_bacc = best_bacc
        print("second cv -> best b_acc: %.3f for pair [%.1f, %d] (%d features) (time: %.3f sec)" % (round(best_bacc, 3),
                                                                                                   best_win, best_wl,
                                                                                                   n_features,
                                                                                                   end - ini))

    def cv_third_step(self, dataset, times, labels, **kwargs):
        use_pca = kwargs.pop("use_pca", True)
        cv_method = kwargs.pop("cv_method", "loo")
        class_method = kwargs.pop("class_method", "centroid")
        n_splits = kwargs.pop("n_splits", 5)
        scale = kwargs.pop("scale", True)
        with_mean = kwargs.pop("with_mean", True)
        n_components = kwargs.pop("n_components", 20)
        dist_method = kwargs.pop("dist_method", "cosine")
        repr_method = kwargs.pop("repr_method", "bopf")

        # kwargs["window"] = self.selected_wins_for_cv
        # kwargs["word_length"] = self.selected_wls_for_cv
        kwargs["verbose"] = True
        if self.all_vectors is None:
            if self["verbose"]:
                print("computing count vectorizers for window = {}, word_length = {}".format(self.selected_wins_for_cv,
                                                                                             self.selected_wls_for_cv),
                      end=" ")
            self.wl_arr, self.win_arr, self.all_vectors = transformer_mp(dataset, times,
                                                                         self.selected_wls_for_cv,
                                                                         self.selected_wins_for_cv,
                                                                         n_process="default",
                                                                         **kwargs)
            # transf = CountVectorizer(**kwargs)
            # transf.fit(dataset, times)
            # self.wl_arr, self.win_arr, _, self.all_vectors = transf.transform(dataset, times)
            if self["verbose"]:
                print("DONE")


        n = len(self.wl_arr)
        vectors_arr = [self.get_concatenated_vectors(self.all_vectors[i]) for i in range(n)]
        ini = time.time()
        bacc_arr = self.multiprocessing_bacc_arr(vectors_arr, self.wl_arr, self.win_arr, n, np.array(labels),
                                                 use_pca=use_pca, cv_method=cv_method,
                                                 class_method=class_method, n_splits=n_splits,
                                                 scale=scale, with_mean=with_mean, n_components=n_components,
                                                 dist_method=dist_method, repr_method=repr_method)
        # for i in range(n):
        #     win = self.win_arr[i]
        #     wl = self.wl_arr[i]
        #     if (win, wl) not in self.selected_tuples:
        #         vectors = self.get_concatenated_vectors(self.all_vectors[i])
        #         real, pred = cv_classify(vectors, np.array(labels), use_pca=use_pca, cv_method=cv_method,
        #                                  class_method=class_method, n_splits=n_splits)
        #         bacc = balanced_accuracy_score(real, pred)
        #         if self["verbose"]:
        #             print("first cv -> pair [%.1f, %d] (%d/%d) ==> balanced_acc:" % (win, wl, i + 1, n),
        #                   round(bacc, 3), end="\r")
        #         bacc_arr.append(bacc)
        end = time.time()

        sorted_best = np.argsort(bacc_arr)[::-1]
        best_bacc = bacc_arr[sorted_best[0]]
        if self.global_best_bacc <= best_bacc:
            self.global_best_bacc = best_bacc
            best_win = self.win_arr[sorted_best[0]]
            best_wl = self.wl_arr[sorted_best[0]]
            best_vectors = self.get_concatenated_vectors(self.all_vectors[sorted_best[0]])
            n_features = best_vectors.shape[1]
            self.selected_tuples.append((best_win, best_wl))
            self.selected_vectors = best_vectors
            print("third cv -> best b_acc: %.3f for pair [%.1f, %d] (%d features) (time: %.3f sec)" % (round(best_bacc,
                                                                                                             3),
                                                                                                       best_win,
                                                                                                       best_wl,
                                                                                                       n_features,
                                                                                                       end - ini))
        else:
            print("accuracy doesnt improve, best combinations are:",
                  self.selected_tuples, " (time: %.3f sec)" % (end-ini))
            self.stop=True

    def cv_third_step_mp(self, dataset, times, labels, **kwargs):
        pass





def cv_pca(count_vectors, labels, n_com_arr, early_stop=5, n_splits=5, class_type="centroid", verbose=True):
    cv = StratifiedKFold(n_splits=n_splits)
    std_scaler = StandardScaler(with_mean=True)
    best_n_com = 0
    best_acc = -1
    c = 0
    for n_com in n_com_arr:
        acc_arr = []
        n_com2 = min(n_com, count_vectors.shape[0], count_vectors.shape[1]-1)
        for train_index, valid_index in cv.split(count_vectors, labels):
            X_train, X_test = count_vectors[train_index], count_vectors[valid_index]
            y_train, y_test = labels[train_index], labels[valid_index]

            if class_type == "centroid":
                X_train, y_train = compute_class_centroids(X_train, y_train)
            elif class_type == "tf_idf":
                X_train, y_train = compute_class_tf_idf(X_train, y_train)

            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)

            pca = PCA(n_components=n_com2)

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            if class_type == "centroid":
                y_pred = predict_by_centroid(X_train, y_train, X_test)
            elif class_type == "tf_idf":
                y_pred = predict_by_tf_idf(X_train, y_train, X_test)
            else:
                raise ValueError("class type '%s' unknown" % class_type)

            acc = balanced_accuracy_score(y_test, y_pred)
            acc_arr.append(acc)
        mean_acc = np.mean(acc_arr)
        if verbose:
            print("mean acc for n_com %d is: " % n_com2, mean_acc)
        if best_acc < mean_acc:
            best_acc = mean_acc
            best_n_com = n_com
            c = 0
        else:
            c += 1

        if c >= early_stop:
            break
    print("best acc: ", best_acc, ", best n_com:", best_n_com)
    return best_n_com
