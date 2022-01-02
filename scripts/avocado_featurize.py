#!/usr/bin/env python
"""Featurize a dataset using avocado"""
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import argparse
from tqdm import tqdm
import time
import avocado
from src.avocado_adapter import Dataset, AVOCADOFeaturizer


def process_chunk(featurizer, chunk, args, verbose=True):
    # Load the reference dataset
    if verbose:
        print("Loading dataset...")
    dataset = Dataset.load(
        args.dataset,
        chunk=chunk,
        num_chunks=args.num_chunks,
    )

    # Featurize the dataset
    if verbose:
        print("Featurizing the dataset...")
    dataset.extract_raw_features(featurizer, keep_models=args.save_models)

    # Save the features.
    if verbose:
        print("Saving the features...")
    dataset.write_raw_features(tag=args.tag)

    if args.save_models:
        # Save the models if desired.
        if verbose:
            print("Saving the models...")
        dataset.write_models(tag=args.tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to featurize.'
    )
    parser.add_argument(
        '--tag',
        default=avocado.settings['features_tag'],
        help='The tag to use for these features. The default is set in '
        'avocado_settings.json. (default: %(default)s)',
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
        'the data at once. This sets the total number of chunks to use. '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=None,
        help='If set, only process this chunk of the dataset. This is '
        'intended to be used to split processing into multiple jobs.'
    )
    parser.add_argument(
        '--save_models',
        action='store_true',
    )
    parser.add_argument("--record_times", type=str, default=None)

    args = parser.parse_args()
    main_ini = time.time()

    # Load the featurizer. For now, we only have the PLAsTiCC featurizer
    # although this could be an option in the future.
    print("Loading featurizer...")
    featurizer = AVOCADOFeaturizer(record_times=args.record_times is not None)

    if args.chunk is not None:
        # Process a single chunk
        process_chunk(featurizer, args.chunk, args)
    else:
        # Process all chunks
        print("Processing the dataset in %d chunks..." % args.num_chunks)
        for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                          dynamic_ncols=True):
            process_chunk(featurizer, chunk, args, verbose=True)

    print("Done!")
    main_end = time.time()
    try:
        out_path = avocado.settings["avocado_directory"]
        f = open(os.path.join(out_path, "log.txt"), "a+")
        f.write("++++++++++++++++++++++++++++++++\n")
        f.write("script_name: avocado_featurize.py\n")
        f.write("dataset: %s\n" % args.dataset)
        f.write("chunk: %s\n" % str(args.chunk))
        f.write("num_chunks: %s\n" % args.num_chunks)
        f.write("execution time: %.3f\n" % (main_end - main_ini))
        f.write("++++++++++++++++++++++++++++++++\n")
        f.close()
    except Exception as e:
        print("failed to write log file, error: ", e)
        try:
            f.close()
        except:
            pass
