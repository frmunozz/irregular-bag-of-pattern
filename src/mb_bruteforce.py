import os
import sys
import numpy as np
import pandas as pd
from dtaidistance import dtw
from .timeseries_object import TimeSeriesObject
import multiprocessing as mp

band_map = {
        0: 'lsstu',
        1: 'lsstg',
        2: 'lsstr',
        3: 'lssti',
        4: 'lsstz',
        5: 'lssty',
    }


def mb_dist(obs1: np.ndarray, obs2: np.ndarray):
    d = 0
    for i in range(len(obs1)):
        if len(obs1[i]) == 0 or len(obs2[i]) == 0:
            # print("warning: empty band")
            continue
        d += dtw.distance_fast(obs1[i], obs2[i])

    return d


def worker(dataset, i_ini, i_end, out_q):
    try:
        print("start worker dtw: ", mp.current_process().name, "for i range [", i_ini, ", ", i_end, "]")
        n = len(dataset)
        rows = np.ones((i_end - i_ini, n)) * -1
        for i in range(i_ini, i_end):
            if (i-i_ini) % 10 == 0:
                print("'%s', i=%d/%d" % (mp.current_process().name, i-i_ini, i_end-i_ini))
            for j in range(n):
                d = mb_dist(dataset[i], dataset[j])
                # print("computed dist")
                rows[i - i_ini][j] = d
        out_q.put((i_ini, i_end, rows))
    except Exception as e:
        print("worker failed with error code:", e)
    finally:
        print("worker '%s' done" % mp.current_process().name)


def mb_dmatrix_mp(dataset, n_process="default"):
    if n_process == "default":
        n_process = mp.cpu_count()

    n = len(dataset)
    n_chunks = 1 + n // n_process
    man = mp.Manager()
    result_queue = man.Queue()
    jobs = []

    for k in range(n_process):
        i_ini = k * n_chunks
        i_end = (k + 1) * n_chunks
        if i_end > n:
            i_end = n
        jobs.append(mp.Process(target=worker,
                               args=(dataset, i_ini, i_end, result_queue)))
        jobs[-1].start()

    for p in jobs:
        p.join()

    dmatrix = np.zeros((n,n))
    j = result_queue.qsize()
    while j > 0:
        i_ini, i_end, rows = result_queue.get()
        dmatrix[i_ini:i_end] = rows
        j -= 1

    return dmatrix
