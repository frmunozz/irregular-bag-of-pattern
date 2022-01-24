# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)


def call1(dataset, num_chunks, args):
    call1_ = "python -m sklearnex fea.1.mmm_bopf_repr_fit.py "
    call1_ += "{0} {1} --timestamp={2} --n_jobs={3} --num_chunks={4}".format(
        dataset, args.config_file, args.timestamp, args.n_jobs, num_chunks
    )
    return call1_


def call2(dataset, num_chunks, args):
    call2_ = "python -m sklearnex fea.2.mmm_bopf_repr_transform.py "
    call2_ += "{0} {1} --timestamp={2} --n_jobs={3} --num_chunks={4}".format(
        dataset, args.config_file, args.timestamp, args.n_jobs, num_chunks
    )
    return call2_


def call3(dataset, num_chunks, args):
    call2_ = "python -m sklearnex fea.3.mmm_bopf_compact_fit.py "
    call2_ += "{0} {1} --timestamp={2} --num_chunks={3}".format(dataset, args.config_file, args.timestamp, num_chunks)
    return call2_


def call4(dataset, num_chunks, args):
    call3_ = "python -m sklearnex fea.4.mmm_bopf_compact_transform.py "
    call3_ += "{0} --timestamp={1} --num_chunks={2}".format(dataset, args.timestamp, num_chunks)
    return call3_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_file",
        help="filename for method IBOPF configuration"
    )
    parser.add_argument(
        '-fit',
        '--fit_dataset',
        default="plasticc_train",
        help='Name of the dataset to fit.',
    )
    parser.add_argument(
        '-transform',
        '--transform_dataset',
        nargs="+",
        default=["plasticc_test", "plasticc_augment"],
        help='List of datasets to Transform.'
    )
    parser.add_argument(
        '-n_chunks',
        '--num_chunks',
        nargs="+",
        default=[1, 100, 20],
        help='The number of chunks to divide each dataset in order. '
             'First num_chunk for fit_dataset, and the rest for transform_dataset',
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        "-c",
        "--compact_method",
        type=str,
        default="LSA",
        help="The compact method to use, options are: LSA or MANOVA"
    )
    parser.add_argument(
        '-n_jobs',
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
    )
    # RUNNING EXAMPLE
    # python fea.pipeline.py optimal_config_lsa.json -fit plasticc_train -transform plasticc_test plasticc_augment_v3 -n_chunks 1 100 10 -c LSA -n_jobs 6

    args = parser.parse_args()
    c = args.compact_method  # LSA MANOVA

    # print("RUNNING 1.mmm_bopf_repr_fit.py for compact_method=%s, dataset=%s" % (c, args.fit_dataset))
    # os.system(call1(args.fit_dataset, args.num_chunks[0], args))
    # for dataset, num_chunks in zip(args.transform_dataset, args.num_chunks[1:]):
    #     print("RUNNING 2.mmm_bopf_repr_transform.py for compact_method=%s, dataset=%s" % (c, dataset))
    #     os.system(call2(dataset, int(num_chunks), args))

    print("RUNNING 3.mmm_bopf_compact_fit.py for compact_method=%s, dataset=%s" % (c, args.fit_dataset))
    os.system(call3(args.fit_dataset, args.num_chunks[0], args))
    for dataset, num_chunks in zip(args.transform_dataset, args.num_chunks[1:]):
        print("RUNNING 4.mmm_bopf_compact_transform.py for compact_method=%s, dataset=%s" % (c, dataset))
        os.system(call4(dataset, num_chunks, args))

    print("DONE!!")
    print("TIMESTAMP: ", args.timestamp)

    # RUNING EXAMPLE
    # python pipeline.py plasticc_train plasticc_test optimal_config_lsa.json --compact_method=LSA --train_num_chunks=1 --test_num_chunks=200
    # python pipeline.py plasticc_train plasticc_test optimal_config_lsa.json --compact_method=LSA --train_num_chunks=1 --test_num_chunks=100 --timestamp=20210916-035944 --n_jobs=6
    # python pipeline.py plasticc_augment_v3 plasticc_test optimal_config_lsa.json --compact_method=LSA --train_num_chunks=10 --test_num_chunks=100 --n_jobs=6

