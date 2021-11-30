# -*- coding: utf-8 -*-
import argparse
import numpy as np

import avocado

import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
from tqdm import tqdm
import time
from collections import defaultdict

"""
this script only work with plasticc_augment
"""


_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def reduce_objects_by_threshold(args):
    train_set = avocado.load(args.train_dataset)
    detections = {}
    classes_by_obj = {}
    for obj in train_set.objects:
        c = np.sum(obj.observations["detected"].to_numpy())
        detections[obj.metadata["object_id"]] = c
        classes_by_obj[obj.metadata["object_id"]] = obj.metadata["class"]

    _selected_objects = []
    _count_objects_per_class = defaultdict(int)
    for k, v in detections.items():
        if v > args.detection_threshold:
            _selected_objects.append(k)
            class_obj = classes_by_obj[k]
            _count_objects_per_class[class_obj] += 1

    return _selected_objects, _count_objects_per_class


def process_chunk(chunk, selected_objs, max_augment_per_class, count_agument, args):

    dataset = avocado.load(args.augment_dataset, metadata_only=False,
                           chunk=chunk, num_chunks=args.num_chunks)

    objects_list = []

    for obj in tqdm(
            dataset.objects, desc="Reading chunk", dynamic_ncols=True, position=0, leave=True
    ):
        ref_obj_id = obj.metadata["reference_object_id"]
        class_obj = obj.metadata["class"]
        if not isinstance(ref_obj_id, str):
            objects_list.append(obj)
        else:
            if ref_obj_id in selected_objs and count_agument[ref_obj_id] < max_augment_per_class[class_obj]:
                objects_list.append(obj)
                count_agument[ref_obj_id] += 1

    augmented_dataset = avocado.Dataset.from_objects(
        "plasticc_augment_v3",
        objects_list,
        chunk=dataset.chunk,
        num_chunks=dataset.num_chunks,
    )

    augmented_dataset.write()

    return count_agument


if __name__ == '__main__':
    ini = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'train_dataset',
        help='Name of the reference object dataset.'
    )
    parser.add_argument(
        'augment_dataset',
        help='Name of the augmented object dataset.'
    )
    parser.add_argument(
        '--detection_threshold',
        type=int,
        default=2,
        help='the minimum number of detection allowed for an object to be used on data augmentation',
    )
    parser.add_argument(
        '--max_objects_per_class',
        type=int,
        default=6000,
        help='the maximum number of augmentation allowed for each object',
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=25,
        help='The dataset will be processed in chunks to avoid loading all of '
             'the data at once. This sets the total number of chunks to use. '
             '(default: %(default)s)',
    )

    args = parser.parse_args()

    count_agument = defaultdict(int)

    selected_objects, count_objects_per_class = reduce_objects_by_threshold(args)
    max_augment_per_class = {k: args.max_objects_per_class // v for k, v in count_objects_per_class.items()}
    print("max augment per class:")
    print(max_augment_per_class)

    # start transforming to MMBOPF repr in chunks
    # Process all chunks
    print("Processing the dataset in %d chunks..." % args.num_chunks)

    for chunk in tqdm(range(args.num_chunks), desc='chunks', dynamic_ncols=True):
        count_agument = process_chunk(chunk, selected_objects, max_augment_per_class, count_agument, args)

    # execution example
    # python plasticc.preprocess_augment_v3.py plasticc_train plasticc_augment
