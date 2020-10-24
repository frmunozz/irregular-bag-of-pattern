import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)
import time
import argparse as ap



if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="name", help="data folder path", required=True, type=str)

    args = parser.parse_args()
    train = main_path + "data/UCRArchive_2018/{}/{}_TRAIN.tsv".format(args.name, args.name)
    test = main_path + "data/UCRArchive_2018/{}/{}_TEST.tsv".format(args.name, args.name)
    n1 = args.n1
    n2 = args.n2
    c = args.c
    n_process = args.n_process

    ini = time.time()
    dmatrix = dmatrix_multiprocessing(in_folder, n1, n2, c, n_process, out_folder)
    end = time.time()

    print("dmatrix shape: ", dmatrix.shape)
    print("execution time (sec): ", end - ini)