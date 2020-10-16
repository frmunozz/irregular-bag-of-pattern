import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bruteforce import dmatrix_multiprocessing
import time


if __name__ == '__main__':
    in_folder = "D:/tesis/tesis/plasticc_sub_dataset/"
    out_folder = "D:/tesis/tesis/bruteforce_dmatrix/"
    n1 = 200
    n2 = 100
    c = 4
    n_process = 6

    ini = time.time()
    dmatrix = dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder)
    end = time.time()

    print("dmatrix shape: ", dmatrix.shape)
    print("execution time (sec): ", end - ini)