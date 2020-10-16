import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bruteforce import dmatrix_multiprocessing
import time
import argparse as ap


if __name__ == '__main__':
    in_folder = "D:/tesis/tesis/data/plasticc_sub_dataset/"
    out_folder = "D:/tesis/tesis/data/bruteforce_dmatrix/"
    parser = ap.ArgumentParser()
    parser.add_argument("-n1", "--nTrain", dest="n1", help="size of train set", required=True, type=int)
    parser.add_argument("-n2", "--nTest", dest="n2", help="size of test set", required=True, type=int)
    parser.add_argument("-c", "--classes", dest="c", help="number of classes", required=True, type=int)
    parser.add_argument("-p", "--nProcess", dest="n_process", help="number of process", required=True, type=int)

    args = parser.parse_args()
    n1 = args.n1
    n2 = args.n2
    c = args.c
    n_process = args.n_process

    ini = time.time()
    dmatrix = dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder)
    end = time.time()

    print("dmatrix shape: ", dmatrix.shape)
    print("execution time (sec): ", end - ini)