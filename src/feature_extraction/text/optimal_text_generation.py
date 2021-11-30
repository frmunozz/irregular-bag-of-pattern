# -*- coding: utf-8 -*-
import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from scipy import sparse
import multiprocessing
from numba import njit
from numba.typed import List
import multiprocessing as mp
import time

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


@njit(cache=True)
def fit_slider(t, n, win, tol):
    # print(list(t))
    secs_i = List()
    secs_j = List()
    secs_ini = List()
    secs_end = List()
    is_forward = List()
    # forward sequences
    j = 0
    i = 0
    while i < n - 1:
        ini = t[i]
        while j < n:
            if t[j] <= ini + win:
                j += 1
            else:
                j -= 1
                break
        if j - i > tol:
            secs_i.append(i)
            secs_j.append(j)
            secs_ini.append(ini)
            secs_end.append(ini + win)
            is_forward.append(True)
        i += 1

    # backward sequences
    # TODO: testing without this
    i = n - 1
    j = n - 1
    while j > 1:
        end = t[j]
        while i > 0:
            if t[i] >= end - win:
                i -= 1
            else:
                i += 1
                break
        if i < 0:
            i = 0

        if j - i > tol:
            secs_i.append(i)
            secs_j.append(j+1)
            secs_ini.append(end - win)
            secs_end.append(end)
            is_forward.append(False)
        j -= 1

    return secs_i, secs_j, secs_ini, secs_end, is_forward


@njit(cache=True)
def get_mean_bp(v, i, j, n, dist="normal"):
    vec = v[i:j]
    # assumes only normal distribution and alphabet size = 4
    # since we are using the cum sum technique it will always be
    # distributed with 0 mean and std 1
    if n == 4:
        return np.array([-0.67, 0, 0.67])
    elif n == 6:
        return np.array([-0.967, -0.43, 0, 0.43, 0.967])
    else:
        # if alphabet is not 4, use uniform distribution
        return np.linspace(np.min(vec), np.max(vec), n + 1)[1:-1]


@njit(cache=True)
def get_var_bp(v, i, j, n):
    vec = v[i:j]
    return np.linspace(0, np.max(vec) - np.min(vec), n+1)[1:-1]


@njit(cache=True)
def get_min_bp(v, i, j, n):
    return get_mean_bp(v, i, j, n)


@njit(cache=True)
def get_max_bp(v, i, j, n):
    return get_mean_bp(v, i, j, n)


@njit(cache=True)
def get_count_bp(v, i, j, n, wl):
    vec = v[i:j]
    c = len(vec)
    _max = c / wl * 2
    return np.linspace(0, _max, n + 1)[1:-1]


@njit(cache=True)
def get_std_bp(v, i, j, n):
    vec = v[i:j]
    inn_vec = np.zeros(2)
    inn_vec[0] = np.min(vec)
    inn_vec[1] = np.max(vec)
    _max = np.std(inn_vec)
    return np.linspace(0, _max, n+1)[1:-1]


@njit(cache=True)
def get_trend_bp(n):
    return np.linspace(-np.pi / 2, np.pi / 2, n+1)[1:-1]


@njit(cache=True)
def get_min_max_bp(v, i, j, n):
    vec = v[i:j]
    diff = np.max(vec) - np.min(vec)
    return np.linspace(0, diff, n+1)[1:-1]


@njit(cache=True)
def forward_segmentator(t, offset, ini, sub_win, n, wl):
    segments = List()
    empty_segments = List()
    empty_segments.append(9999)
    i = 0
    j = 1
    ini_time = ini
    end_time = ini + sub_win
    for k in range(wl):
        while i < n - 1 and t[i] < ini_time:
            i += 1
        i_within_range = end_time > t[i] >= ini_time
        while j < n and t[j] <= end_time:
            j += 1
        j_within_range = end_time >= t[j] > ini_time
        if i == n or not i_within_range or not j_within_range:
            segments.append([-1, -1])
            empty_segments.append(k)
        else:
            segments.append([i + offset, j + offset])
        ini_time += sub_win
        end_time += sub_win
        i = j

    return segments, empty_segments


@njit(cache=True)
def backward_segmentator(t, offset, end, sub_win, n, wl):
    segments = List()
    empty_segments = List()
    empty_segments.append(9999)
    i = n-2
    j = n-1
    ini_time = end - sub_win
    end_time = end
    for k in range(wl):
        while i > 0 and t[i] >= ini_time:
            i -= 1
        i += 1  # set i as internal on the sub window
        i_within_range = end_time > t[i] >= ini_time
        while j > 1 and t[j] > end_time:
            j -= 1
        j_within_range = end_time >= t[j] > ini_time

        if j <= 0 or not i_within_range or not j_within_range:
            segments.append([-1, -1])
            empty_segments.append(k)
        else:
            segments.append([i + offset, j + offset])
        ini_time -= sub_win
        end_time -= sub_win
        j = i

    return segments, empty_segments


@njit(cache=True)
def main_segmentator(times, i, j, ini, end, is_forward, win, wl):
    sub_win = win / wl
    offset = i
    t = times[i:j]
    n = len(t)

    if wl == 1:
        segments = List()
        segments.append([i, j])
        empty_segments = List()
        empty_segments.append(wl + 1000)
    else:
        if is_forward:
            segments, empty_segments = forward_segmentator(t, offset, ini, sub_win, n, wl)
        else:
            segments, empty_segments = backward_segmentator(t, offset, end, sub_win, n, wl)

    return segments, empty_segments


@njit(cache=True)
def segmentator(times, ii, jj, ini, win, wl):
    """ only time based segmentation """
    sub_win = win / wl
    # print("sub window:", sub_win)
    offset = ii
    t = times[ii:jj]
    n = len(t)

    if wl == 1:
        segments = List()
        segments.append([ii, jj])
        empty_segments = List()
        empty_segments.append(wl + 1000)
    else:
        _i = 0
        _j = 1
        segments = List()
        empty_segments = List()
        empty_segments.append(wl + 1000)
        ini_time = ini
        end_time = ini + sub_win
        for _kk in range(wl):
            while _i < n-1 and t[_i] < ini_time:
                _i += 1
            while _j < n and t[_j] <= end_time:
                _j += 1
            if _i == n or t[_i] >= end_time or t[_j-1] < ini_time:
                segments.append([-1, -1])
                empty_segments.append(_kk)
            else:
                segments.append([_i + offset, _j + offset])
            ini_time += sub_win
            end_time += sub_win
            _i = _j
    return segments, empty_segments


@njit(cache=True)
def cum_sum(v, n):
    cum1 = np.zeros(n + 1)
    cum2 = np.zeros(n + 1)
    psum = 0
    psum2 = 0
    for j in range(n):
        psum += v[j]
        psum2 += v[j] ** 2
        cum1[j + 1] = psum
        cum2[j + 1] = psum2
    return cum1, cum2


@njit(cache=True)
def digitize(x, bins):
    if x < bins[0]:
        return 0
    n = len(bins)
    for i in range(1, n):
        if x < bins[i]:
            return i
    return n


@njit(cache=True)
def mean_sigma(cum1, cum2, i, j):
    sumx = cum1[j] - cum1[i]
    sumx2 = cum2[j] - cum2[i]
    meanx = sumx / (j - i)
    meanx2 = sumx2 / (j - i)
    sigmax = np.sqrt(np.abs(meanx2 ** 2 - meanx ** 2))
    return meanx, sigmax


@njit(cache=True)
def compute_mean(cum1, i, j, meanx, sigmax, mean_bp):
    if j == i:
        return -1, False
    sumsub = cum1[j] - cum1[i]
    avgsub = sumsub / (j - i)
    if sigmax > 0:
        mean = (avgsub - meanx) / sigmax
    else:
        mean = 0
    return digitize(mean, mean_bp), True


@njit(cache=True)
def compute_min(v, i, j, min_bp):
    y = v[i:j]
    if len(y) == 0:
        # we cant compute min
        return -1, False
    elif len(y) == 1:
        _min = y[0]
    else:
        _min = np.min(y)
    return digitize(_min, min_bp), True


@njit(cache=True)
def compute_max(v, i, j, max_bp):
    y = v[i:j]
    if len(y) == 0:
        # we cant compute min
        return -1, False
    elif len(y) == 1:
        _max = y[0]
    else:
        _max = np.max(y)
    return digitize(_max, max_bp), True


@njit(cache=True)
def compute_count(v, i, j, count_bp):
    y = v[i:j]
    c = len(y)
    return digitize(c, count_bp), True


@njit(cache=True)
def compute_trend(v, t, i, j, trend_bp):
    y = v[i:j]
    x = t[i:j]
    if len(x) < 2:
        # we cant compute trend
        return -1, False
    X = x - x.mean()
    Y = y - y.mean()
    xy = X.dot(Y)
    xx = X.dot(X)
    if xx != 0:
        slope = (X.dot(Y)) / (X.dot(X))
        trend = np.arctan(slope)
    else:
        trend = np.pi / 2  # the equivalent of arctan(infinity)
    return digitize(trend, trend_bp), True


@njit(cache=True)
def compute_std(v, i, j, std_bp):
    if len(v[i:j]) < 2:
        # we cant compute std
        return -1, False
    else:
        std_value = np.std(v[i:j])
    return digitize(std_value, std_bp), True


@njit(cache=True)
def compute_var(v, i, j, var_bp):
    y = v[i:j]
    if len(y) < 2:
        return -1, False
    else:
        var_value = np.var(y)
    return digitize(var_value, var_bp), True


@njit(cache=True)
def compute_min_max(v, i, j, min_max_bp):
    y = v[i:j]
    if len(y) < 2:
        # we cant compute min_max
        return -1, False
    else:
        min_max = np.max(y) - np.min(y)
    return digitize(min_max, min_max_bp), True


@njit(cache=True)
def segment_to_char(v, t, _ini, _end, tol, wl, quantity, cum1,
                    i, j, meanx, sigmax, mean_bp, trend_bp, std_bp,
                    min_max_bp, var_bp, min_bp, max_bp, count_bp, alphabet_size, alpha_size_full):
    if _end - _ini >= max(1, tol // wl):
        val = 0
        weight = 1
        nn = len(quantity)
        for q in range(nn):
            q_str = quantity[nn - q - 1]
            if q_str == "mean":
                # compute mean
                iter_val, is_valid = compute_mean(cum1, _ini, _end, meanx, sigmax, mean_bp)
            elif q_str == "trend":
                # compute trend (slope)
                iter_val, is_valid = compute_trend(v, t, _ini, _end, trend_bp)
            elif q_str == "std":
                # compute std
                iter_val, is_valid = compute_std(v, _ini, _end, std_bp)
            elif q_str == "min_max":
                # compute the min_max
                iter_val, is_valid = compute_min_max(v, _ini, _end, min_max_bp)
            elif q_str == "var":
                # compute variance
                iter_val, is_valid = compute_var(v, _ini, _end, var_bp)
            elif q_str == "min":
                # compute min value
                iter_val, is_valid = compute_min(v, _ini, _end, min_bp)
            elif q_str == "max":
                # compute max valur
                iter_val, is_valid = compute_max(v, _ini, _end, max_bp)
            elif q_str == "count":
                iter_val, is_valid = compute_count(v, _ini, _end, count_bp)

            else:
                iter_val = -1
                is_valid = False
                print("unknown quantity ", q_str)

            if is_valid:
                val += weight * iter_val
                weight *= alphabet_size[nn - q - 1]
            else:
                # the segment is not valid, we break the for and set val to the special character
                # we should never reach this code
                return alpha_size_full, False
    else:
        # the segment was empty or with only 1 element
        # we set val to the special character
        return alpha_size_full, False
    return val, True


@njit(cache=True)
def generate_word(v, t, tol, wl, quantity, cum1, cum2,
                  i, j, mean_bp, trend_bp, std_bp, min_max_bp,
                  var_bp, min_bp, max_bp, count_bp,
                  alphabet_size, segments):
    meanx, sigmax = mean_sigma(cum1, cum2, i, j)
    _invalid_chars = 0
    word = np.zeros(wl)
    alpha_size_full = alphabet_size.prod()
    for k in range(wl):
        _ini, _end = segments[k]
        ####################################
        # SEGMENT TO CHAR
        ####################################
        val, is_valid = segment_to_char(v, t, _ini, _end, tol, wl, quantity, cum1, i, j,
                                        meanx, sigmax, mean_bp, trend_bp, std_bp, min_max_bp,
                                        var_bp, min_bp, max_bp, count_bp,
                                        alphabet_size, alpha_size_full)
        if is_valid:
            word[k] = int(((alpha_size_full + 1) ** k) * val)
        else:
            word[k] = int(((alpha_size_full + 1) ** k) * val)
            _invalid_chars += 1

    return word, _invalid_chars


@njit(cache=True)
def get_alpha_by_quantity(alphabet_size, quantities):
    alpha_by_quantity = {}
    ll = len(alphabet_size)
    for ii in range(ll):
        _x = quantities[ii]
        _y = alphabet_size[ii]
        alpha_by_quantity[_x] = _y
    return alpha_by_quantity


@njit(cache=True)
def generate_univariate_text(v, t, n, bop_size, _alph_size_by_quantity,
                             win, wl, alphabet_size, quantity, num_reduction, tol, mean_bp_dist,
                             threshold, secs_i, secs_j, secs_ini, secs_end, is_forward, cum1, cum2):
    doc = np.zeros(bop_size)
    dropped_count = 0
    mean_bp = np.zeros(alphabet_size[0])  # hardcoded, should use alpha by Q
    trend_bp = np.zeros(alphabet_size[0])
    std_bp = np.zeros(alphabet_size[0])
    min_max_bp = np.zeros(alphabet_size[0])
    var_bp = np.zeros(alphabet_size[0])
    min_bp = np.zeros(alphabet_size[0])
    max_bp = np.zeros(alphabet_size[0])
    count_bp = np.zeros(alphabet_size[0])

    if wl == 1:
        # setting global break points for all possible quantitites
        if "mean" in _alph_size_by_quantity:
            mean_bp = get_mean_bp(v, 0, n, _alph_size_by_quantity["mean"], dist=mean_bp_dist)
        if "std" in _alph_size_by_quantity:
            std_bp = get_std_bp(v, 0, n, _alph_size_by_quantity["std"])
        if "trend" in _alph_size_by_quantity:
            trend_bp = get_trend_bp(_alph_size_by_quantity["trend"])
        if "min_max" in _alph_size_by_quantity:
            min_max_bp = get_min_max_bp(v, 0, n, _alph_size_by_quantity["min_max"])
        if "var" in _alph_size_by_quantity:
            var_bp = get_var_bp(v, 0, n, _alph_size_by_quantity["var"])
        if "min" in _alph_size_by_quantity:
            min_bp = get_min_bp(v, 0, n, _alph_size_by_quantity["min"])
        if "max" in _alph_size_by_quantity:
            max_bp = get_max_bp(v, 0, n, _alph_size_by_quantity["max"])
        if "count" in _alph_size_by_quantity:
            count_bp = get_count_bp(v, 0, n, _alph_size_by_quantity["count"], wl)

    _prev_word = -1  # used for numerosity reduction
    n_secs = len(secs_i)
    for _K in range(n_secs):
        i = secs_i[_K]
        j = secs_j[_K]
        ini = secs_ini[_K]
        end = secs_end[_K]
        _is_forward = is_forward[_K]

        if j - i >= max(1, tol):
            ####################################
            # SEGMENTATION
            ####################################
            segments, empty_segments = main_segmentator(t, i, j, ini, end, _is_forward, win, wl)

            # old version of segmentator deprecated
            # segments, empty_segments = segmentator(t, i, j, ini, win, wl)

            # print(i, j, ini, "segments:", list(segments))

            l_e = len(empty_segments) - 1
            if wl - threshold < l_e:
                # too many segments are empty
                cond1 = False
                word_p = 0
                _invalid_chars = 0
                # print(i, j, -1, _invalid_chars, empty_segments, segments)
            else:
                # we can transform the sequence to word

                if wl > 1:
                    # local break points
                    if "mean" in _alph_size_by_quantity:
                        mean_bp = get_mean_bp(v, i, j, _alph_size_by_quantity["mean"], dist=mean_bp_dist)
                    if "trend" in _alph_size_by_quantity:
                        trend_bp = get_trend_bp(_alph_size_by_quantity["trend"])
                    if "std" in _alph_size_by_quantity:
                        std_bp = get_std_bp(v, i, j, _alph_size_by_quantity["std"])
                    if "min_max" in _alph_size_by_quantity:
                        min_max_bp = get_min_max_bp(v, i, j, _alph_size_by_quantity["min_max"])
                    if "var" in _alph_size_by_quantity:
                        var_bp = get_var_bp(v, i, j, _alph_size_by_quantity["var"])
                    if "min" in _alph_size_by_quantity:
                        min_bp = get_min_bp(v, i, j, _alph_size_by_quantity["min"])
                    if "max" in _alph_size_by_quantity:
                        max_bp = get_max_bp(v, i, j, _alph_size_by_quantity["max"])
                    if "count" in _alph_size_by_quantity:
                        count_bp = get_count_bp(v, i, j, _alph_size_by_quantity["count"], wl)

                ####################################
                # GENERATE WORD
                ####################################

                word, _invalid_chars = generate_word(v, t, tol, wl, quantity, cum1, cum2, i, j,
                                                     mean_bp, trend_bp, std_bp, min_max_bp,
                                                     var_bp, min_bp, max_bp, count_bp,
                                                     alphabet_size, segments)
                # print(i, j, word, _invalid_chars, empty_segments, segments)
                word_p = word.sum()
                cond1 = len(word) > _invalid_chars

            if cond1:
                # process word
                # support only special character
                word_p_int = int(word_p)  # word pointer in the final Bag-of-Word vector
                if num_reduction and _prev_word != word_p_int and word_p_int <= bop_size:
                    doc[word_p_int] += 1
                    _prev_word = word_p_int
            else:
                dropped_count += 1
        else:
            dropped_count += 1

    return doc, dropped_count

##
##
##
##
##
##
@njit(cache=True)
def generate_multivariate_text(fluxes, times, bands_ini, bands_end, n_bands, bands_lbls, bop_size, _alph_size_by_quantity,
                               win, wl, alphabet_size, quantity, num_reduction, tol, mean_bp_dist,
                               threshold):

    if threshold is None:
        threshold = wl

    doc_mb = np.zeros(bop_size * n_bands)
    dropped_mb = 0
    all_none = True
    for _i in range(n_bands):
        k = bands_lbls[_i]
        # print("BAND ", k)

        if bands_ini[_i] != -1 and bands_end[_i] != -1:
            v = fluxes[bands_ini[_i]:bands_end[_i]+1]
            t = times[bands_ini[_i]:bands_end[_i] + 1]
            n = len(v)
            n_copy = n
            cum1, cum2 = cum_sum(v, n)
            ####################################
            # FIT SLIDER
            ####################################
            secs_i, secs_j, secs_ini, secs_end, is_forward = fit_slider(t, n_copy, win, tol)
            # print(secs_i)
            # print(secs_j)
            # print(secs_ini)
            # print("N segments: ", len(secs_i))

            ####################################
            # TRANSFORM
            ####################################
            doc, dropped_count = generate_univariate_text(v, t, n, bop_size, _alph_size_by_quantity,
                                                          win, wl, alphabet_size, quantity,
                                                          num_reduction, tol, mean_bp_dist,
                                                          threshold, secs_i, secs_j, secs_ini,
                                                          secs_end, is_forward, cum1, cum2)
            # print(dropped_count)
            doc_mb[_i * bop_size: (_i + 1) * bop_size] = doc
            dropped_mb += dropped_count

            if all_none and np.sum(doc) > 0:
                all_none = False
        else:
            doc_mb[_i * bop_size: (_i + 1) * bop_size] = np.zeros(bop_size)
    if all_none:
        return None, dropped_mb

    return doc_mb, dropped_mb


def transform(dataset, n_bands, win=50, wl=4, alphabet_size=np.array([4]),
              quantity=np.array(["mean"]), num_reduction=True,
              tol=6, mean_bp_dist="normal", threshold=None):

    _alpha_size_by_quantity = get_alpha_by_quantity(alphabet_size, quantity)
    bop_size = (alphabet_size.prod() + 1) ** wl  # only work with special character
    m = len(dataset)

    if threshold is None:
        threshold = wl

    corpus = []

    for i in tqdm(range(m), dynamic_ncols=True,
                  desc="[win: %.3f, wl: %d]" % (win, wl)):
        fluxes, times, bands_ini, bands_end = dataset[i]
        doc, _dropped_count = generate_multivariate_text(fluxes, times, bands_ini, bands_end, n_bands, _BANDS, bop_size,
                                                         _alpha_size_by_quantity, win, wl,
                                                         alphabet_size, quantity, num_reduction,
                                                         tol, mean_bp_dist, threshold)
        # doc is a sparse matrix of shape (1, N_bands * bop_size)
        # Bag-of-Pattern should be a sparse matrix of shape (N, N_bands * bop_size)
        # then we will create a list of all the doc and then apply vertical_stack
        doc = sparse.csr_matrix(doc)
        if doc is not None:
            corpus.append(doc)
        else:
            # if failed, we append an empty sparse matrix
            corpus.append(sparse.csr_matrix((1, n_bands * bop_size)))

    # then we create the final sparse matrix
    m_bop = sparse.vstack(corpus, format="csr")

    return m_bop


def multiprocess_text_transform(dataset, n_bands, win=50, wl=4, alphabet_size=np.array([4]),
                                quantity=np.array(["mean"]), num_reduction=True,
                                tol=6, mean_bp_dist="normal", threshold=None, n_jobs=-1,
                                extra_desc="", chunk_size=10, position=0, leave=True):
    _alpha_size_by_quantity = get_alpha_by_quantity(alphabet_size, quantity)
    bop_size = (alphabet_size.prod() + 1) ** wl  # only work with special character

    if threshold is None:
        threshold = wl

    mp_class = MPTextTransform(n_bands, win, wl, alphabet_size, quantity, num_reduction,
                               tol, mean_bp_dist, threshold, bop_size, _alpha_size_by_quantity)
    print("class defined, starting transform")
    m_bop = mp_class.transform(dataset, n_jobs, extra_desc=extra_desc, chunk_size=chunk_size,
                               position=position, leave=leave)
    return m_bop


def mp_worker(sub_dataset, n_bands, win, wl, alphabet_size,
              quantity, num_reduction,
              tol, mean_bp_dist, threshold, res_queue, ini, end, worker_id):
    try:
        _alpha_size_by_quantity = get_alpha_by_quantity(alphabet_size, quantity)
        bop_size = (alphabet_size.prod() + 1) ** wl  # only work with special character
        m = len(sub_dataset)

        if threshold is None:
            threshold = wl

        corpus = []

        for i in range(m):
            fluxes, times, bands_ini, bands_end = sub_dataset[i]
            doc, _dropped_count = generate_multivariate_text(fluxes, times, bands_ini, bands_end, n_bands, np.array(_BANDS),
                                                             bop_size,
                                                             _alpha_size_by_quantity, win, wl,
                                                             alphabet_size, quantity, num_reduction,
                                                             tol, mean_bp_dist, threshold)
            # doc is a sparse matrix of shape (1, N_bands * bop_size)
            # Bag-of-Pattern should be a sparse matrix of shape (N, N_bands * bop_size)
            # then we will create a list of all the doc and then apply vertical_stack
            if doc is not None:
                doc = sparse.csr_matrix(doc)
                corpus.append(doc)
            else:
                # if failed, we append an empty sparse matrix
                corpus.append(sparse.csr_matrix((1, n_bands * bop_size)))

        # then we create the final sparse matrix
        m_bop = sparse.vstack(corpus, format="csr")

        res_queue.put((ini, end, worker_id, m_bop))
    except Exception as e:
        print("worker (id: %d)[%d,%d] failed, error: %s" % (worker_id, ini, end, e))


def mp_text_transform(dataset, n_bands, win=50, wl=4, alphabet_size=np.array([4]),
                      quantity=np.array(["mean"]), quantity_symbol="", num_reduction=True,
                      tol=6, mean_bp_dist="normal", threshold=None, n_jobs=-1, print_time=False):
    if isinstance(alphabet_size, list):
        alphabet_size = np.array(alphabet_size)
    if isinstance(quantity, list):
        quantity = np.array(quantity)
    # print("alphabet_size:", alphabet_size, type(quantity))
    # print("quantity", quantity, type(quantity))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    # print("n_jobs", n_jobs)

    print("[win: %.3f, wl: %d, Q: %s]: PARALLEL PROCESSING (%d JOBS)... " % (win, wl, quantity_symbol, n_jobs), end="\r")
    process_ini_time = time.time()

    N = len(dataset)
    sub = N // n_jobs
    m = mp.Manager()
    result_queue = m.Queue()
    jobs = []
    for i in range(n_jobs):
        ini = i * sub
        end = (i+1) * sub
        if i == n_jobs - 1:
            end = N
        sub_dataset = dataset[ini:end]
        jobs.append(mp.Process(target=mp_worker,
                               args=(sub_dataset, n_bands, win, wl, alphabet_size,
                                     quantity, num_reduction, tol, mean_bp_dist, threshold, result_queue,
                                     ini, end, i)))
        jobs[-1].start()

    for p in jobs:
        p.join()

    results_bop = [None] * n_jobs
    num_res = result_queue.qsize()
    while num_res >0:
        ini, end, worker_id, m_bop = result_queue.get()
        results_bop[worker_id] = m_bop
        num_res -= 1

    m_final_bop = sparse.vstack(results_bop, format="csr")
    process_end_time = time.time()
    if print_time:
        print("[win: %.3f, wl: %d, Q: %s]: PARALLEL PROCESSING (%d JOBS)... DONE (%.3f secs)" %
              (win, wl, quantity_symbol, n_jobs, process_end_time - process_ini_time))
    # print("[win: %.3f, wl: %d, Q: %s]: PARALLEL PROCESSING (%d JOBS)... DONE! (time: %f)" % (
    #     win, wl, quantity_symbol, n_jobs, process_end_time - process_ini_time), end="\r")
    return m_final_bop, process_end_time - process_ini_time


class MPTextTransform:
    def __init__(self, n_bands, win, wl, alphabet_size, quantity, num_reduction,
                 tol, mean_bp_dist, threshold, bop_size, _alpha_size_by_quantity):
        self.N_bands = n_bands
        self.win = win
        self.wl = wl
        self.alpha = alphabet_size
        self.Q = quantity
        self.num_reduction = num_reduction
        self.tol = tol
        self.mean_bp_dist = mean_bp_dist
        self.threshold = threshold
        self.bop_size = bop_size
        self._alpha_by_Q = _alpha_size_by_quantity

    def worker(self, x):
        fluxes, times, bands_ini, bands_end = x
        print(bands_ini, bands_end)
        doc, _dropped_count = generate_multivariate_text(fluxes, times, bands_ini, bands_end, self.N_bands,
                                                         np.array(_BANDS), self.bop_size, self._alpha_by_Q, self.win,
                                                         self.wl, self.alpha, self.Q, self.num_reduction,
                                                         self.tol, self.mean_bp_dist, self.threshold)
        doc2 = sparse.csr_matrix(doc)
        return doc2

    def transform(self, dataset, n_jobs=-1, extra_desc="", chunk_size=1, position=0, leave=True):
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count() + 4
            print("n_jobs", n_jobs)
        print("entering process_map")
        print("X", type(dataset), len(dataset))
        r = process_map(self.worker, dataset, max_workers=n_jobs,
                        desc="[win: %.3f, wl: %d%s]" % (self.win, self.wl, extra_desc),
                        chunksize=chunk_size, position=position, leave=leave)
        m_bop = sparse.vstack(r, format="csr")
        return m_bop
