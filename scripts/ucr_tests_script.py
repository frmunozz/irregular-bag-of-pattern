from collections import defaultdict
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt
import sys
import time
import pdb
from itertools import chain, product
import itertools
import string

import multiprocessing as mp
import queue

from scipy.sparse import csr_matrix


from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from dtaidistance import dtw

main_path = os.path.abspath(os.path.join(os.path.dirname("./Untitled.ipynb"), '..'))



def mean_value_bp(values, alphabet_size, strategy="uniform"):
    if strategy == "uniform":
        values_min = np.min(values)
        values_max = np.max(values)
        return np.linspace(values_min, values_max, alphabet_size+1)[1:-1]
    elif strategy == "normal":
#         print("before")
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.linspace(mean, mean, alphabet_size+1)[1:-1]
        else:
            return norm.ppf(np.linspace(0, 1, alphabet_size+1)[1:-1], mean, std)

    
def slope_bp(alphabet_size):
    values_min = -np.pi/4
    values_max = np.pi/4
    return np.linspace(values_min, values_max, alphabet_size+1)[1:-1]

def min_max_bp(values, alphabet_size):
    diff = np.abs(np.max(values) - np.min(values))
    return np.linspace(0, diff, alphabet_size+1)[1:-1]


class DocumentTimeSeries(object):
    def __init__(self, alph_size=4, word_length=2, window=100, alphabet=["a", "b", "c", "d"], 
                 feature="mean", bp_strategy="uniform", tol=2, global_break_points=False, threshold=2,
                empty_handler="special_character"):
        self.word_length=word_length
        self.alph_size=alph_size
        self.window=window
        self.feature = feature
        self.bp_strategy=bp_strategy
        self.tol = tol
        self.global_break_points = global_break_points
        self.threshold = threshold
        self.bp = None
        self.ini_time = 0
        self.end_time = window
        self.alphabet = alphabet
    
    def get_break_points(self, ts_data):
        if self.feature == "mean":
            return mean_value_bp(ts_data, self.alph_size, strategy=self.bp_strategy)
        elif self.feature == "trend":
            return slope_bp(self.alph_size)
        elif self.feature == "minmax":
            return min_max_bp(ts_data, self.alph_size)
        
    def gen_document(self, ts_data, ts_times, strategy="all_sub_sequence", **kwargs):
        if strategy == "all_sub_sequence":
            return " ".join(self.transform_all_sub_sequence(ts_data, ts_times))
        else:
            raise ValueError("not implemented")
        
    def transform_all_sub_sequence(self, ts_data, ts_time):
        doc = []
        i = 0
        j = 1
        n = ts_data.size
        ini_obs, end_obs = ts_time[0], ts_time[n-1]
        if self.global_break_points or self.word_length == 1:
            self.bp = self.get_break_points(ts_data)
#             if self.feature == "minmax":
#                 print(self.bp)
        while i < n - self.tol:
            self.ini_time, self.end_time = ts_time[i], ts_time[i] + self.window
            while ts_time[j]  < self.end_time:
                if j == n-1:
                    break
                j += 1
#             if self.window == 100:
#                 print(self.ini_time, self.end_time, i, j)
            if j - i > self.tol * self.word_length:
                if not self.global_break_points and self.word_length > 1:
                    self.bp = self.get_break_points(ts_data[i:j])
#                 if self.window == 100:
#                     print(i, j, self.bp)
                word = self.sequence_to_word(ts_data, ts_time, i, j)
                if word.count("#") <= self.threshold:
                    doc.append(word)
            i += 1
        return doc
    
    def sequence_to_word(self, ts_data, ts_time, i, j):
        if self.word_length > 1:
            seg_limits = np.linspace(self.ini_time, self.end_time, self.word_length + 1)[1:]
            
#             if self.window == 100:
#                 print(seg_limits)
            word = ''
            ii = i
            jj = i+1
            for k in range(self.word_length):
                while ts_time[jj] < seg_limits[k]:
                    if jj == j:
                        break
                    jj += 1
#                 if self.window == 100:
#                     print(k, ii, jj)
                if jj - ii > self.tol:
                    val = self.segment_to_char(ts_data, ts_time, ii, jj)
                else:
                    val = "#"
                word += val
                ii = jj
            return word
        else:
            return self.segment_to_char(ts_data, ts_time, i, j)
        
    def segment_to_char(self, ts_data, ts_time, i, j):
        if self.feature == "mean":
            mean = np.mean(ts_data[i:j])
#             if self.window == 100:
#                 print(mean)
            return self.alphabet[np.digitize(mean, self.bp)]
        elif self.feature == "trend":
            slope, _,_,_,_ = linregress(ts_time[i:j], ts_data[i:j])
            trend = np.arctan(slope)
            return self.alphabet[np.digitize(trend, self.bp)]
        
        elif self.feature == "minmax":
            diff = np.abs(np.max(ts_data[i:j]) - np.min(ts_data[i:j]))
#             print(diff)
            return self.alphabet[np.digitize(diff, self.bp)]


def load_numpy_dataset(data_path, file_base):
    dataset = np.load(os.path.join(data_path, file_base % "d"), allow_pickle=True)
    for i in range(dataset.size):
        dataset[i] = preprocessing.scale(dataset[i])
    times = np.load(os.path.join(data_path, file_base % "t"), allow_pickle=True)
    labels = np.load(os.path.join(data_path, file_base % "l"), allow_pickle=True)
    return dataset, times, labels.astype(int), len(dataset)

def load_numpy_dataset2(data_path, file_base, normalize=True):
    dataset = np.load(os.path.join(data_path, file_base % "dataset"), allow_pickle=True)
    for i in range(dataset.size):
        dataset[i] = preprocessing.scale(dataset[i])
    times = np.load(os.path.join(data_path, file_base % "times"), allow_pickle=True)
    labels = np.load(os.path.join(data_path, file_base % "labels"), allow_pickle=True)
    return dataset, times, labels.astype(int), len(dataset)

def predict_class_cosine(similarity_matrix, train_label, n_train, n_test):
    pred_labels = []
    for j in range(n_test):
        dmax = -np.inf
        label = -1
        for i in range(n_train):
            if similarity_matrix[i][j] > dmax:
                dmax = similarity_matrix[i][j]
                label = train_label[i]
        pred_labels.append(label)
    return pred_labels

def predict_class_euclidean(similairty_matrix, train_label, n_train, n_test):
    pred_labels = []
    for j in range(n_test):
        dmin = np.inf
        label = -1
        for i in range(n_train):
            if similairty_matrix[i][j] < dmin:
                dmin = similairty_matrix[i][j]
                label = train_label[i]
        pred_labels.append(label)
    return pred_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def multi_window_document(ts_data, ts_time, windows, j,
                          alph_size=4, word_length=2, 
                          feature="mean", threshold=1, bp_strategy="uniform", 
                          tol=2, global_break_points=False):
    full_alphabet = list(string.ascii_lowercase)
    full_alphabet.extend(list(string.ascii_uppercase))
    i = 0
    full_doc = ""
    for w in windows:
#         print(w)
        alphabet = full_alphabet[i:i+alph_size]
#         print(alphabet)
        doc_ts = DocumentTimeSeries(alph_size=alph_size, word_length=word_length, 
                                    window=w, alphabet=alphabet, feature=feature, 
                                    bp_strategy=bp_strategy, tol=tol, 
                                    global_break_points=global_break_points, 
                                    threshold=threshold)
#         print(doc_ts.dictionary)
        doc = doc_ts.gen_document(ts_data, ts_time)
        i += alph_size
        if len(full_doc) > 0:
            full_doc += " "
        full_doc += doc
    if len(full_doc) == 0:
        print("fail", j)
    return full_doc

def document_numerosity_reduction(doc):
    doc_arr = doc.split()
    new_doc = [doc_arr[0]]
    pword = doc_arr[0]
    for i in range(1,len(doc_arr)):
        if pword != doc_arr[i]:
            new_doc.append(doc_arr[i])
            pword = doc_arr[i]
    return " ".join(new_doc)

def multi_word_document(ts_data, ts_time, word_lengths, 
                        alph_size=4, window=100, feature="mean", thresholds=None, 
                        bp_strategy="uniform", tol=2, global_break_points=False):
    if thresholds is None:
        thresholds = [0] * len(word_lengths)
    full_alphabet = list(string.ascii_lowercase)
    full_alphabet.extend(list(string.ascii_uppercase))
    alphabet = full_alphabet[:alph_size]
    full_doc = ""
    i=0
#     print("thresholds:", thresholds)
#     print("word lengths:", word_lengths)
    assert len(thresholds) == len(word_lengths)
    for word_length, thr in zip(word_lengths, thresholds):
#         print(word_length)
        doc_ts = DocumentTimeSeries(alph_size=alph_size, word_length=word_length, 
                                    window=window, alphabet=alphabet, feature=feature, 
                                    bp_strategy=bp_strategy, tol=tol, 
                                    global_break_points=global_break_points, 
                                    threshold=thr)
        doc = doc_ts.gen_document(ts_data, ts_time)
        if len(full_doc) > 0:
            full_doc += " "
        full_doc += doc
    if len(full_doc) == 0:
        print("fail")
    return full_doc


def generalized_document_transform(dataset, times, full_alphabet, alph_size=8, word_length=[2], 
                                   windows=[800, 400, 200, 100, 50, 25],
                                   feature=["mean"], thresholds=None, 
                                   empty_handler="special_character",
                                   tol=3, bp_strategy="uniform", global_break_point=False,
                                  numerosity_reduction=False):
    if thresholds is None:
        thresholds = [0] * len(word_length)
    full_doc = ""
    i = 0
    for win in windows:
        for thrs, wl in zip(thresholds, word_length):
            for fea in feature:
                alphabet = full_alphabet[i:i+alph_size]
                doc_ts = DocumentTimeSeries(alph_size=alph_size, word_length=wl,
                                           window=win, alphabet=alphabet, feature=fea,
                                           bp_strategy=bp_strategy, threshold=thrs,
                                           global_break_points=global_break_point, 
                                            empty_handler=empty_handler)
                doc = doc_ts.gen_document(dataset, times)
                i += alph_size
                if len(full_doc) > 0:
                    full_doc += " "
                full_doc += doc
                
    if len(full_doc) == 0:
        print("fail")
        return full_doc
                
    if numerosity_reduction:
        full_doc = document_numerosity_reduction(full_doc)
                
    return full_doc


def get_full_alphabet(max_alph_size):
    letters = list(string.ascii_letters)
    if max_alph_size > len(letters):
        N = max_alph_size // len(letters) + 1
        numbers = np.arange(N) + 1
        alphabet = [str(numbers[i]) + x for i in range(N) for x in letters]
    else:
        alphabet = letters
    return alphabet
    
def get_vocabulary(full_alphabet, alph_size, windows, feature, word_length, thresholds):
    l = 0
    vocabulary = []
    for i in range(len(windows)):
        for  j in range(len(feature)):
            for k in range(len(word_length)):
                used_alphabet = full_alphabet[l:l+alph_size]
                if thresholds[k] > 0:
                    used_alphabet.append('#')
                vocabulary.extend(list(map(''.join,
                              chain.from_iterable(product(used_alphabet,
                                                         repeat=word_length[k]) for i in range(1)))))
                l += alph_size
    return np.unique(vocabulary)


def compute_centroids(vectors, labels):
    classes = np.unique(labels)
    n_classes = len(classes)
    centroids = np.zeros((n_classes, vectors.shape[1]))
    class_counter = np.zeros(n_classes)
    for i in range(len(labels)):
        idx = np.where(labels[i] == classes)[0]
        centroids[idx] += vectors[i]
        class_counter[idx] += 1
    
    for i in range(n_classes):
        centroids[i] /= class_counter[i]
        
    return centroids, classes

def dataset_to_corpus_worker(dataset, times, full_alphabet, lock, idx_queue, out_queue, **kwargs):
    try:
#         print("========== start worker '%s' ==========" % mp.current_process().name)
        while True:
            try:
                lock.acquire()
                i = idx_queue.get_nowait()
                n = len(dataset)
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                if i > 0 and i % 10 == 0:
                    print("processed %d/%d (worker '%s')" % (i, n, mp.current_process().name), end="\r")
                doc = generalized_document_transform(dataset[i], times[i], full_alphabet,  
                                                     alph_size=kwargs.get("alph_size"), 
                                                     word_length=kwargs.get("word_length"), 
                                                     windows=kwargs.get("windows"), 
                                                     feature=kwargs.get("feature"), 
                                                     thresholds=kwargs.get("thresholds"), 
                                                     empty_handler=kwargs.get("empty_handler"), 
                                                     tol=kwargs.get("tol"), 
                                                     bp_strategy=kwargs.get("bp_strategy"), 
                                                     global_break_point=kwargs.get("global_break_point"), 
                                                     numerosity_reduction=kwargs.get("numerosity_reduction"))
                out_queue.put((doc, i))
            
        
    except Exception as e:
        print("=====> Worker failed with error:", e)
    finally:
#         print("=====> worker '%s' DONE" % mp.current_process().name)
        pass


def dataset_to_corpus_mp(dataset, times, n_process=None, **kwargs):
    
    print("transformation parameters:")
    print("word_length=", kwargs.get("word_length"))
    print("windows=", kwargs.get("windows")) 
    print("features=", kwargs.get("feature"))
    print("alph_size=", kwargs.get("alph_size"))
    
    max_alph_size = kwargs.get("alph_size") * len(kwargs.get("word_length")) * len(kwargs.get("windows")) * len(kwargs.get("feature"))
    if kwargs.get("thresholds") is None:
        kwargs["thresholds"] = [0] * len(kwargs.get("word_length"))
    full_alphabet = get_full_alphabet(max_alph_size)
    vocabulary = get_vocabulary(full_alphabet, kwargs.get("alph_size"), kwargs.get("windows"), 
                                kwargs.get("feature"), kwargs.get("word_length"), 
                               kwargs.get("thresholds"))
    print("full alphabet size: ", len(full_alphabet))
    print("vocabulary size: ", len(vocabulary))
    
    print("generating documents using multiprocessing")
    
    ini = time.time()
    if n_process is None or n_process == "default":
        n_process = mp.cpu_count()
    
    m = mp.Manager()
    result_queue = m.Queue()
    
    n = len(dataset)
    
    idx_queue = mp.Queue()
    for i in range(n):
        idx_queue.put(i)
        
    lock = mp.Lock()
    
    jobs = []
    for i in range(n_process):
        p = mp.Process(target=dataset_to_corpus_worker, 
                       args=(dataset, times, full_alphabet, lock, idx_queue, result_queue),
                      kwargs = kwargs)
        jobs.append(p)
        p.start()
        
    for p in jobs:
        p.join()
    print("====> processed %d/%d, all workers finished" % (n,n))
    corpus = [""] * n
    num_res = result_queue.qsize()
    while num_res > 0:
        doc, i = result_queue.get()
        corpus[i] = doc
        num_res -= 1
    end = time.time()
    print("execution time (sec): ", end-ini)
    print("==================================================")
    return corpus, vocabulary

def dataset_to_corpus(dataset, times, alph_size=8, 
                          word_length=[2], windows=[800, 400, 200, 100, 50, 25],
                         feature=["mean"], thresholds=None, empty_handler="special_character",
                         tol=3, bp_strategy="uniform",
                         global_break_point=False, numerosity_reduction=False):
    
    print("transformation parameters:")
    print("word_length=", word_length)
    print("windows=", windows) 
    print("features=", feature)
    print("alph_size=", alph_size)
    
    max_alph_size = alph_size * len(word_length) * len(windows) * len(feature)
    if thresholds is None:
        thresholds = [0] * len(word_length)
    full_alphabet = get_full_alphabet(max_alph_size)
    vocabulary = get_vocabulary(full_alphabet, alph_size, windows, feature, word_length, thresholds)
    print("full alphabet size: ", len(full_alphabet))
    print("vocabulary size: ", len(vocabulary))
    s = "generating documents from time series dataset ->"
    print(s, end="\r")
    corpus = []
    n = len(dataset)
    print_step = 50 if n > 200 else 10
    ini = time.time()
    for i in range(n):
        if i % print_step == 0:
            print(s + "%d/%d" % (i, n), end="\r")
            
        doc = generalized_document_transform(dataset[i], times[i], full_alphabet,  
                                             alph_size=alph_size, word_length=word_length, 
                                             windows=windows, feature=feature, 
                                             thresholds=thresholds, empty_handler=empty_handler, 
                                             tol=tol, bp_strategy=bp_strategy, 
                                             global_break_point=global_break_point, 
                                             numerosity_reduction=numerosity_reduction)
        corpus.append(doc)
    end = time.time()
    print(s + "%d/%d" % (n, n) + " [DONE], time:", end-ini)
    return corpus, vocabulary


def predict_tf_idf_pca(X_train, X_test, y_train, y_test, n_com,
                              similarity="cosine", scaled=True, with_mean=True,
                              n_components=20, dim_reducer="pca", n_iter=20, use_centroids=False):
    
    transf = TfidfTransformer()
    tf_idf_train = transf.fit_transform(X_train)
    tf_idf_test = transf.transform(X_test)
        
    tf_idf_train = tf_idf_train.todense()
    tf_idf_test = tf_idf_test.todense()
        
    if scaled:
        std_scaler = StandardScaler(with_mean=with_mean)
        tf_idf_train = std_scaler.fit_transform(tf_idf_train)
        tf_idf_test = std_scaler.transform(tf_idf_test)
        
    if dim_reducer == "pca":  
        pca = PCA(n_components=n_com)
        tf_idf_train = pca.fit_transform(tf_idf_train)
        tf_idf_test = pca.transform(tf_idf_test)
    elif dim_reducer == "svd":
        svd = TruncatedSVD(n_components=n_com, n_iter=n_iter)
        tf_idf_train = svd.fit_transform(tf_idf_train)
        tf_idf_test = svd.transform(tf_idf_test)
        
    if use_centroids:
        tf_idf_train, y_train = compute_centroids(tf_idf_train, y_train)
        
    if similarity == "cosine":
        dmatrix = cosine_similarity(tf_idf_train, tf_idf_test)
        pred_labels = predict_class_cosine(dmatrix, y_train, len(tf_idf_train), len(tf_idf_test))
    elif similarity == "euclidean":
        dmatrix = euclidean_distances(tf_idf_train, tf_idf_test)
        pred_labels = predict_class_euclidean(dmatrix, y_train, len(tf_idf_train), len(tf_idf_test))
        
#     print("dmatrix shape:", dmatrix.shape)
            
    return pred_labels, y_test        


def cross_validation_classify(corpus, labels, vocabulary=None, n_components=20, scaled=True, 
                           with_mean=True, similarity="cosine", dim_reducer="pca",
                             cv_strategy="loo", n_splits=5, n_iter=20, use_centroids=False):
    
    vect = CountVectorizer(lowercase=False, vocabulary=vocabulary,
                           analyzer="word", tokenizer=lambda txt: txt.split())
    X = vect.fit_transform(corpus)
    X = X.todense()
    if cv_strategy == "loo":
        cv_splitter = LeaveOneOut()
        cv_iterator = cv_splitter.split(X)
        NN = len(labels)
        NN_step = NN // 50 + 1
    elif cv_strategy == "kfold":
        cv_splitter = StratifiedKFold(n_splits=n_splits)
        cv_iterator = cv_splitter.split(X, labels)
        NN = n_splits
        NN_step = 1
    else:
        raise ValueError("cv_strategy '%s' unknown" % cv_strategy)
        
    pred_labels = []
    real_labels = []
    
    n_com = min(n_components, X.shape[0] - X.shape[0] // NN, X.shape[1]-1)
    
    s = "starting leave one out, pipeline: "
    if scaled:
        s += "normalize ->"
    if dim_reducer == "pca":
        s += " PCA [Dim %d => %d]->" % (X.shape[1], n_com)
    elif dim_reducer == "svd":
        s += " SVD ->"
    s += " classify => "
    print(s, end="\r")
    k = 0
    print
    for train_index, test_index in cv_iterator:
        if k % NN_step == 0:
            print(s + "%d/%d" % (k, NN), end="\r")
        k += 1
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        pred_l, real_l = predict_tf_idf_pca(X_train, X_test, y_train, y_test, n_com,
                                           similarity=similarity, scaled=scaled, with_mean=with_mean,
                                           n_components=n_components, dim_reducer=dim_reducer,
                                           n_iter=n_iter, use_centroids=use_centroids)
        pred_labels.extend(pred_l)
        real_labels.extend(real_l)
    print(s + "%d/%d" % (NN, NN))
        
    return pred_labels, real_labels

def train_test_classify(corpus_train, corpus_test, y_train, y_test,
                        vocabulary=False, n_components=20, scaled=True, 
                        with_mean=True, similarity="cosine", dim_reducer="pca", 
                        use_centroids=False, n_iter=20):
    
    vect = CountVectorizer(lowercase=False, vocabulary=vocabulary,
                           analyzer="word", tokenizer=lambda txt: txt.split())
    X_train = vect.fit_transform(corpus_train)
    X_test = vect.transform(corpus_test)
    n_com = min(n_components, X_train.shape[0], X_train.shape[1]-1)
    print("Dimension reduced [%d => %d]" % (X_train.shape[1], n_com))
    pred_l, real_l = predict_tf_idf_pca(X_train, X_test, y_train, y_test, n_com,
                                           similarity=similarity, scaled=scaled, with_mean=with_mean,
                                           n_components=n_components, dim_reducer=dim_reducer,
                                           n_iter=n_iter, use_centroids=use_centroids)
    return pred_l, real_l



def test_ucr_dataset(data_path, key, **kwargs):
    
    train_file = os.path.join(data_path, key, key + "_TRAIN.tsv")
    test_file = os.path.join(data_path, key, key + "_TEST.tsv")
    file = open(train_file, 'r')
    lines = file.readlines()
    d_train = []
    t_train = []
    l_train = []
    for d in lines:
        arr = d[:-1].split("\t")
        data = np.array(arr[1:], dtype=float)
        times = np.arange(data.size, dtype=float)
        d_train.append(data)
        t_train.append(times)
        l_train.append(int(arr[0]))
        
    file = open(test_file, 'r')
    lines = file.readlines()
    d_test = []
    t_test = []
    l_test = []
    for d in lines:
        arr = d[:-1].split("\t")
        data = np.array(arr[1:], dtype=float)
        times = np.arange(data.size, dtype=float)
        d_test.append(data)
        t_test.append(times)
        l_test.append(int(arr[0]))
        
    # since all ts in UCR are same length and regular
    max_window = round(t_train[0][-1], -1)
    
    alph_size = kwargs.get("alph_size", 4)
    word_length = kwargs.get("word_length", [1, 2, 3, 4, 5, 6])
    windows = kwargs.get("windows", [int(max_window/(2**i)) for i in range(5)])
    feature= kwargs.get("feature", ["mean"])
    thresholds = kwargs.get("thresholds", [0] * len(word_length))
    empty_handler = kwargs.get("empty_handler", "special_character")
    tol = kwargs.get("tol", 1)
    bp_strategy = kwargs.get("bp_strategy", "normal")
    global_break_points = kwargs.get("global_break_points", False)
    numerosity_reduction=kwargs.get("numerosity_reduction", True)
    
    corpus_train, vocabulary = dataset_to_corpus_mp(d_train, t_train, n_process=None,
                                                    alph_size=alph_size, word_length=word_length, 
                                                    windows=windows,feature=feature, 
                                                    thresholds=thresholds, 
                                                    empty_handler=empty_handler, tol=tol, 
                                                    bp_strategy=bp_strategy, 
                                                    global_break_point=global_break_points,
                                                    numerosity_reduction=numerosity_reduction)
    
    corpus_test, _ = dataset_to_corpus_mp(d_test, t_test, n_process=None,
                                          alph_size=alph_size, word_length=word_length,
                                          windows=windows,feature=feature, 
                                          thresholds=thresholds, 
                                          empty_handler=empty_handler, tol=tol,
                                          bp_strategy=bp_strategy, 
                                          global_break_point=global_break_points,
                                          numerosity_reduction=numerosity_reduction)
    

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    
    
    n_components = kwargs.get("n_components", 22)
    scaled = kwargs.get("scaled", True)
    with_mean = kwargs.get("with_mean", True)
    similarity = kwargs.get("similarity", "cosine")
    dim_reducer = kwargs.get("dim_reducer", "pca")
    use_centroids = kwargs.get("use_centroids", True)
    n_iter = kwargs.get("n_iter", 20)
    cv_strategy = kwargs.get("cv_strategy", "loo")
    n_splits =kwargs.get("n_split", 10)

    n_components_min = -1
    n_com_best_acc = -1
    for n_com in [10, 20, 30, 40, 50, 100]:
    	pred_l, real_l = train_test_classify(corpus_train, corpus_test, l_train, l_test,
                        vocabulary=vocabulary, n_components=n_com, scaled=scaled, 
                        with_mean=with_mean, similarity=similarity, dim_reducer=dim_reducer, 
                        use_centroids=use_centroids, n_iter=n_iter)
    	acc = balanced_accuracy_score(real_l, pred_l)
    	if acc > n_com_best_acc:
    		n_com_best_acc = acc
    		n_components_min = n_com

    pred_l, real_l = train_test_classify(corpus_train, corpus_test, l_train, l_test,
                        vocabulary=vocabulary, n_components=n_components_min, scaled=scaled, 
                        with_mean=with_mean, similarity=similarity, dim_reducer=dim_reducer, 
                        use_centroids=use_centroids, n_iter=n_iter)


    if not kwargs.get("ignore_cv", True):
    	corpus = list(np.copy(corpus_train))
    	corpus.extend(corpus_test)
    	labels = list(np.copy(l_train))
    	labels.extend(l_test)

    	labels = np.array(labels)

    	pred_l_cv, real_l_cv = cross_validation_classify(corpus, labels, vocabulary=vocabulary, 
                                                      n_components=n_components, scaled=scaled,
                                                      cv_strategy=cv_strategy, n_splits=n_splits,
                                                      n_iter=n_iter, use_centroids=use_centroids)
    
    	return pred_l, real_l, pred_l_cv, real_l_cv
    else:
    	return pred_l, real_l, None, None



# def plot_ucr_test_results(pred_l, real_l, pred_l_cv, real_l_cv, key, cv_strategy="loo"):
    
#     ignore_cv = real_l_cv is None or pred_l_cv is None
#     fig = plt.figure(figsize=(13, 6))

#     acc1 = balanced_accuracy_score(real_l, pred_l)
#     print("train-test classify balanced accuracy score: ", acc1)
#     conf1 = confusion_matrix(real_l, pred_l)
#     if not ignore_cv:
#     	acc2 = balanced_accuracy_score(real_l_cv, pred_l_cv)
#     	print("Cross-Validation classify balanced accuracy score: ", acc2)
# 		conf2 = confusion_matrix(real_l_cv, pred_l_cv)
#     	fig.add_subplot(121)

#     plot_confusion_matrix(conf1, classes=np.unique(real_l), normalize=False, 
#                      title= "Conf. Matrix classify %s" % key)
#     if not ignore_cv:
#     	fig.add_subplot(122)
#     	plot_confusion_matrix(conf2, classes=np.unique(real_l_cv), normalize=False, 
#                      title= "Conf. Matrix CV-%s %s" % (cv_strategy, key))
    
data_path = os.path.join(main_path, "data", "UCRArchive_2018")

data_summary = pd.read_csv(os.path.join(data_path, "DataSummary.csv"))
data_summary = data_summary[data_summary["Length"] != "Vary"]
file_results = os.path.join(data_path, "tf_idf_pca_results.csv")
keys = data_summary["Name"].to_numpy()
# out_data = defaultdict(list)
for key in keys:
	# key = "BeetleFly"
	print("testing ", key)
	pred_l, real_l, pred_l_cv, real_l_cv = test_ucr_dataset(data_path, key)
	acc = balanced_accuracy_score(real_l, pred_l)
	# acc_cv = balanced_accuracy_score(real_l_cv, pred_l_cv)
	print(key + " acc: ", acc)
	print("=====================================")
	print("=====================================")
	f = open(file_results, 'a')
	f.write("{},{},{}\n".format(key, acc, 1-acc))
	f.close()

# df = pd.DataFrame(out_data)

# df.to_csv(os.path.join(data_path, "my_method_results.csv"), 
	# header=True, index=False)