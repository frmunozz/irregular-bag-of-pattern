import argparse
from tqdm import tqdm
import FATS

import avocado

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def raw_featurize(objects):
    list_raw_features = []
    object_ids = []
    for obj in tqdm(objects, desc="Object", dynamic_ncols=True):
        for b in obj.bands:
            data = obj.observations
            # TODO
