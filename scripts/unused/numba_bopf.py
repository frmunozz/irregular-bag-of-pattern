# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import avocado

import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, main_path)
from src.timeseries_object import FastIrregularUTSObject
from src.feature_extraction.text.text_generation import TextGeneration
from src.feature_extraction.window_slider import TwoWaysSlider
from src.feature_extraction.text.optimal_text_generation import get_alpha_by_quantity, generate_multivariate_text
from src.feature_extraction.text.count_words import multivariate_count_words_flattened
from scipy import sparse

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]

if __name__ == '__main__':
    N_bands = 6
    fluxes = np.array([])
    times = np.array([])
    bands_ini = []
    bands_end = []
    objs = {}
    word_length = 3
    window_width = 50
    alpha = np.array([4, 4, 4])
    Q = np.array(["mean", "trend", "std"])
    thrs = 2
    last_end = -1
    for iii in range(N_bands):
        ttt = np.random.random(50) * 100
        ttt = np.unique(ttt)
        yyy = np.random.random(len(ttt))
        fluxes = np.append(fluxes, yyy)
        times = np.append(times, ttt)
        bands_ini.append(last_end + 1)
        bands_end.append(len(fluxes)-1)
        last_end = len(fluxes)-1
        objs[_BANDS[iii]] = FastIrregularUTSObject(yyy, ttt)

    bands_ini = np.array(bands_ini)
    bands_end = np.array(bands_end)

    print(bands_ini, bands_end)

    a_by_Q = get_alpha_by_quantity(alpha, Q)
    bop_size = (alpha.prod() + 1) ** word_length  # only work with special character
    print("bop_size:", bop_size)

    if thrs is None:
        thrs = word_length

    ini = time.time()
    doc_mb, dropped_mb = generate_multivariate_text(fluxes, times, bands_ini, bands_end, N_bands, np.array(_BANDS),
                                                    bop_size, a_by_Q,
                                                    window_width, word_length, alpha, Q, True, 3, "normal",
                                                    thrs)
    end = time.time()
    print(dropped_mb)
    print(type(doc_mb))
    print(doc_mb.shape)
    print(doc_mb.sum())

    print("TIME FIRST EXECUTION NUMBA:", end - ini)

    # ini = time.time()
    # doc_mb, dropped_mb = generate_multivariate_text(v_ts, t_ts, N_bands, bop_size, a_by_Q,
    #                                                 window_width, word_length, alpha, Q, True, 3, "normal",
    #                                                 thrs)
    # end = time.time()
    # print("TIME SECOND EXECUTION NUMBA:", end - ini)
    # ini = time.time()
    # doc_mb, dropped_mb = generate_multivariate_text(v_ts, t_ts, N_bands, bop_size, a_by_Q,
    #                                                 window_width, word_length, alpha, Q, True, 3, "normal",
    #                                                 thrs)
    # end = time.time()
    # print("TIME third EXECUTION NUMBA:", end - ini)

    ini = time.time()
    doc_gen = TextGeneration(window_width, word_length, alphabet_size=alpha,
                             quantity=Q, tol=3, threshold=thrs, mean_bp_dist="uniform")
    slider = TwoWaysSlider(window_width, tol=3)
    doc_mb, dropped_mb = doc_gen.transform_object(objs, slider)
    end = time.time()
    print("TIME FIRST EXECUTION NOT-NUMBA:", end - ini)
    print(dropped_mb)
    doc_mb2 = multivariate_count_words_flattened([doc_mb], _BANDS, bop_size)
    print(type(doc_mb2))
    print(doc_mb2.shape)
    print(doc_mb2.sum())

    # ini = time.time()
    # doc_gen = TextGeneration(window_width, word_length, alphabet_size=alpha,
    #                          quantity=Q, tol=3, threshold=thrs, mean_bp_dist="uniform")
    # slider = TwoWaysSlider(window_width, tol=3)
    # doc_mb, dropped_mb = doc_gen.transform_object(objs, slider)
    # end = time.time()
    # print("TIME SECOND EXECUTION NOT-NUMBA:", end - ini)