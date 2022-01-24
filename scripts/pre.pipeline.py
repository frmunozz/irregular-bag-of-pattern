# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from ibopf.preprocesing import get_mmbopf_plasticc_path


def call1(args, C, timestamp):
    call1 = "python -m sklearnex pre.1.find_best_quantities.py "
    call1 += "{0} --compact_method={1} --alpha={2} --max_power={3} --timestamp={4} --n_jobs={5}".format(
        args.dataset, C, args.alpha, args.max_power, timestamp, args.n_jobs
    )
    return call1


def call2(C, timestamp, quantity_search_out_path):
    call2 = "python pre.2.rank_best_quantities.py {0} {1} {2}".format(
        C, timestamp, quantity_search_out_path
    )
    return call2


def call3(args, C, timestamp, resume_file):
    call3 = "python -m sklearnex pre.3.find_best_config.py {0} {1} ".format(args.dataset, resume_file)
    call3 += "--top_k={0} --resolution_max={1} --alpha={2} --compact_method={3} --timestamp={4} --n_jobs={5}".format(
        args.top_k, args.resolution_max, args.alpha, C, timestamp, args.n_jobs
    )
    return call3


def resume_file(quantity_search_out_path):
    return os.path.join(quantity_search_out_path, "quantity_search_resume.csv")


def quantity_search_path(main_path, c):
    return os.path.join(main_path, "quantity_search", c.lower())


def execute_pipleline(args, timestamp):

    if args.alpha ** args.max_power <= 4 ** 6:
        raise ValueError("need to specify a power higher than 4^6")

    main_path = get_mmbopf_plasticc_path()
    # quantity_search_out_path = os.path.join(main_path, "quantity_search")
    # resume_file = os.path.join(quantity_search_out_path, "quantity_search_resume.csv")
    c = args.compact_method
    print('RUNNING pre.1.find_best_quantities.py for %s' % c)
    os.system(call1(args, c, timestamp))
    print('RUNNING pre.2.rank_best_quantities.py for %s' % c)
    os.system(call2(c, timestamp, quantity_search_path(main_path, c)))
    print("RUNNING pre.3.find_best_config.py for %s" % c)
    os.system(call3(args, c, timestamp, resume_file(quantity_search_path(main_path, c))))

    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to find best combination of quantities on.'
    )
    parser.add_argument(
        "--compact_method",
        type=str,
        default="LSA",
        help="The compact method to use, options are: LSA or MANOVA"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=4,
        help="alphabet size to use during the search"
    )

    parser.add_argument(
        "--max_power",
        type=int,
        default=8,
        help="max power for the vocabulary expressed as V=alpha^(power)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="the top K single-resolution representations to try on this multi-resolution search"
    )
    parser.add_argument(
        "--resolution_max",
        type=int,
        default=4,
        help="The maximum number of resolutions to include in the optimal multi-resolution combination"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
    )

    args = parser.parse_args()

    timestamp = args.timestamp

    execute_pipleline(args, timestamp)

    # RUNING EXAMPLE
    # python pre.pipeline.py plasticc_train --compact_method=MANOVA --n_jobs=6
    # python pre.pipeline.py plasticc_train --n_jobs=6
    # python pre.pipeline.py plasticc_train --timestamp=20210914-031605 --n_jobs=6
