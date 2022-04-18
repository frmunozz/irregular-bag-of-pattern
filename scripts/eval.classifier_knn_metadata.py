# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import sys
import time
from sklearn.metrics import balanced_accuracy_score
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from ibopf.pipelines.method import IBOPF
from ibopf.preprocesing import get_ibopf_plasticc_path

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'train_dataset',
        help='Name of the train dataset to train on.'
    )
    parser.add_argument(
        'test_dataset',
        help='Name of the test dataset to train on.'
    )

    parser.add_argument(
        "config_file",
        help="filename for method IBOPF configuration"
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        "-nc",
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )
    parser.add_argument(
        "-m",
        '--use_metadata',
        default=None,
        type=str,
        help="Use the optional metadata on the classifier"
    )

    parser.add_argument(
        "-p",
        '--prototype',
        default=None,
        type=str,
        help="Use prototypes to compute the KNN classifier"
    )

    parser.add_argument(
        "-n"
        '--normalizer',
        default=None,
        type=str,
        help="Use Normalizer on the pipeline"
    )

    parser.add_argument(
        "-s"
        '--scaler',
        default=None,
        type=str,
        help="Use StandarScaler on the pipeline"
    )
    args = parser.parse_args()
    main_ini = time.time()

    # get files and folders
    print("GETTING DIRECTORIES AND FILES...")

    main_path = get_ibopf_plasticc_path()
    # get representation folder (plasticc/MMBOPF/representation)
    repr_directory = os.path.join(main_path, "representation")

    # get compact representation folder (plasticc/MMBOPF/compact)
    compact_directory = os.path.join(main_path, "compact")

    # get compact test representation folder (plasticc/MMBOPF/compact/plasticc_test_compact_timestamp)
    test_compact_directory = os.path.join(compact_directory, "%s_compact_%s" % (args.test_dataset, args.timestamp))

    # get classification folder (plasticc/MMBOPF/classification)
    classification_directory = os.path.join(main_path, "classification")
    # check if exists
    if not os.path.exists(classification_directory):
        os.mkdir(classification_directory)

    print("GETTING COMPACT REPRESENTATIONS...")
    # get compact representation for train set and labels
    train_compact = np.load(os.path.join(compact_directory, "%s_compact_%s.npy" % (args.train_dataset, args.timestamp)))
    train_labels = np.load(os.path.join(repr_directory, "%s_labels_%s.npy" % (args.train_dataset, args.timestamp)))

    print("GETTING PIPELINE BASED ON CONFIGURATION FILE...")
    # load method configuration
    method = IBOPF()
    method.config_from_json(os.path.join(main_path, args.config_file))
    method.print_ssm_time = True

    # get pipeline for classifier
    classes = np.unique(train_labels)
    pipeline = method.get_classification_pipeline(classes)

    print("TRAIN CLASSIFIER FOR %s... " % args.train_dataset, end="")
    # train(fit) classifier on compact representation of train set
    ini = time.time()
    pipeline.fit(train_compact, train_labels)
    end = time.time()
    print("DONE (time: %.3f secs)" % (end - ini))

    # predict(transform) classes on compact representation of test set in chunks
    print("PREDICT CLASSES FOR %s IN CHUNKS..." % args.test_dataset)
    test_labels = np.load(os.path.join(repr_directory, "%s_labels_%s.npy" % (args.test_dataset, args.timestamp)))
    print("test labels", test_labels[0], test_labels[1], test_labels[-1])
    pred_labels = np.array([])
    for chunk_id in range(args.num_chunks):
        print("PREDICT CHUNK %d... " % chunk_id, end="")
        f = os.path.join(test_compact_directory, "compact_chunk_%d.npy" % chunk_id)
        test_compact_i = np.load(f)
        ini_i = time.time()
        pred_i = pipeline.predict(test_compact_i)
        end_i = time.time()
        print("DONE (time: %.3f secs)" % (end_i - ini_i))
        pred_labels = np.append(pred_labels, pred_i)

    # save predicted labels
    print("SAVE PREDICTED LABELS...")
    pred_labels_file = os.path.join(classification_directory, "%s_%s_predict_%s" % (
        args.train_dataset, args.test_dataset, args.timestamp))
    np.save(pred_labels_file, pred_labels)

    # print prediction accuracy
    test_labels_known = test_labels[test_labels < 99]
    pred_labels_known = pred_labels[pred_labels < 99]
    print("len test labels (label <99):", test_labels_known.shape)
    print("len pred labels (label <99):", pred_labels_known.shape)
    bacc = balanced_accuracy_score(test_labels_known, pred_labels_known)
    print("balanced accuracy of 1NN-classification: ", bacc)
    main_end = time.time()

    # save log
    logfile = os.path.join(compact_directory, "log.txt")
    f = open(logfile, "a+")
    f.write("\n**** EXECUTED 4.mmm_bopf_classification.py *****\n")
    f.write(".... train_dataset: %s\n" % args.train_dataset)
    f.write(".... test_dataset: %s\n" % args.test_dataset)
    f.write(".... config_file: %s\n" % args.config_file)
    f.write(".... timestamp: %s\n" % args.timestamp)
    f.write(".... balanced accuracy: %.3f\n" % bacc)
    f.write("==== execution time: %.3f sec\n" % (main_end - main_ini))
    f.write("*****************************************\n")
    f.close()

    print("FINISHED!!")
