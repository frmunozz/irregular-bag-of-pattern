import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bruteforce import dmatrix_multiprocessing
import time
import argparse as ap


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-path", dest="path", help="data folder path", required=True, type=str)
    parser.add_argument("-n1", "--nTrain", dest="n1", help="size of train set", required=True, type=int)
    parser.add_argument("-n2", "--nTest", dest="n2", help="size of test set", required=True, type=int)
    parser.add_argument("-c", "--classes", dest="c", help="number of classes", required=True, type=int)
    parser.add_argument("-p", "--nProcess", dest="n_process", help="number of process", required=True, type=int)
    parser.add_argument("-distType", dest="dist_type", help="distance funcion name to use", required=False, type=str)

    args = parser.parse_args()
    in_folder = args.path + "plasticc_subsets/scenario1_ratio_2-8/"
    out_folder = args.path + "bruteforce_dmatrix/scenario1_ratio_2-8/"
    n1 = args.n1
    n2 = args.n2
    c = args.c
    n_process = args.n_process

    dist_type = "dtw"
    if args.dist_type:
        dist_type = args.dist_type

    ini = time.time()
    dmatrix = dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder, dist_type=dist_type)
    end = time.time()

    print("dmatrix shape: ", dmatrix.shape)
    print("execution time (sec): ", end - ini)