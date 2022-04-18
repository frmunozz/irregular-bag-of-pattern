import numpy as np
import pandas as pd
from .timeseries_object import TimeSeriesObject
import avocado
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from avocado.utils import AvocadoException
import os
from .settings import settings


def get_avocado_settings(name):
    return settings[name]


def get_ibopf_plasticc_path():
    directory = settings["IBOPF"]["directory"]
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def gen_dataset(df: pd.DataFrame, df_meta: pd.DataFrame):
    n = df_meta.shape[0]
    res = np.zeros(n, dtype=object)
    labels = np.zeros(n, dtype=object)
    for i in range(df_meta.shape[0]):
        metadata = dict(df_meta.iloc[i])
        df_i = df[df["object_id"] == metadata["object_id"]]
        df_i = df_i.sort_values(by=["mjd"])
        res[i] = TimeSeriesObject(df_i, **metadata)
        labels[i] = metadata["target"]
    return res, labels


def gen_dataset_from_h5(filename, bands, verify_input_chunks=False, num_folds=5, select_survey=None):
    dataset = avocado.load(
        filename,
        verify_input_chunks=verify_input_chunks,
    )
    res = []
    labels = []
    fold_id = []
    all_folds = label_folds(dataset.metadata, num_folds=num_folds)
    for reference_object in tqdm(
            dataset.objects, desc="Object", dynamic_ncols=True
    ):
        condition = True
        if select_survey == "ddf":
            condition = reference_object.metadata["ddf"] == 1
        elif select_survey == "wdf":
            condition = reference_object.metadata["ddf"] == 0

        if condition:
            labels.append(reference_object.metadata["class"])
            res.append(TimeSeriesObject.from_astronomical_object(reference_object).fast_format_for_numba_code(bands))
            fold_id.append(all_folds[reference_object.metadata["object_id"]])
    return np.array(res), np.array(labels), dataset.metadata, np.array(fold_id, dtype=int)


def single_band_dataset(multi_band_dataset: np.ndarray, band):
    res = np.zeros(multi_band_dataset.size, dtype=object)
    for i in range(multi_band_dataset.size):
        res[i] = multi_band_dataset[i]._single_band_sequence(band)
    return res


def label_folds(metadata, num_folds=None, random_state=None):
    """Separate the dataset into groups for k-folding
        This is only applicable to training datasets that have assigned
        classes.
        If the dataset is an augmented dataset, we ensure that the
        augmentations of the same object stay in the same fold.
        Parameters
        ----------
        num_folds : int (optional)
            The number of folds to use. Default: settings['num_folds']
        random_state : int (optional)
            The random number initializer to use for splitting the folds.
            Default: settings['fold_random_state'].
        Returns
        -------
        fold_indices : `pandas.Series`
            A pandas Series where each element is an integer representing the
            assigned fold for each object.
    """
    if num_folds is None:
        num_folds = settings["num_folds"]

    if random_state is None:
        random_state = settings["fold_random_state"]

    if "class" not in metadata:
        raise AvocadoException(
                "Dataset does not have labeled classes! Can't separate "
                "into folds."
        )

    if "reference_object_id" in metadata:
            # We are operating on an augmented dataset. Use original objects to
            # determine the folds.
        is_augmented = True
        reference_mask = metadata["reference_object_id"].isna()
        reference_metadata = metadata[reference_mask]
    else:
        is_augmented = False
        reference_metadata = metadata

    reference_classes = reference_metadata["class"]
    folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
    )
    fold_map = {}
    for fold_number, (fold_train, fold_val) in enumerate(
        folds.split(reference_classes, reference_classes)
    ):
        for object_id in reference_metadata.index[fold_val]:
            fold_map[object_id] = fold_number

    if is_augmented:
        fold_indices = metadata["reference_object_id"].map(fold_map)
        fold_indices[reference_mask] = metadata.index.to_series().map(fold_map)
    else:
        fold_indices = metadata.index.to_series().map(fold_map)

    fold_indices = fold_indices.astype(int)

    return fold_indices


def rearrange_splits(fold_ids):
    folds = np.unique(fold_ids)
    fold_splits = []
    for i in folds:
        test_split = np.where(fold_ids == i)[0]
        train_split = np.where(fold_ids != i)[0]
        fold_splits.append((train_split, test_split))
    return fold_splits


# def get_small_dataset(dataset, times, labels, sort_class_count_index, classes, m, n1, n2, c):
#     # n: size of the small dataset
#     # c: use the 'c' classes most present in the original dataset
#
#     classes_cut = []
#     for i in sort_class_count_index[:c]:
#         classes_cut.append(classes[i])
#
#     data_time_tuple = []
#     labels_cut = []
#     for t, y, l in zip(times, dataset, labels):
#         if l in classes_cut:
#             data_time_tuple.append((t, y))
#             labels_cut.append(l)
#
#     x_train, x_test, y_train, y_test = train_test_split(data_time_tuple, labels_cut,
#                                                         test_size=len(labels_cut) - n1,
#                                                         random_state=0,
#                                                         stratify=labels_cut)
#
#     dataset_split = []
#     times_split = []
#     for d_tuple in x_train:
#         times_split.append(d_tuple[0])
#         dataset_split.append(d_tuple[1])
#
#     _, x_test, _, y_test = train_test_split(x_test, y_test,
#                                             test_size=n2,
#                                             random_state=0,
#                                             stratify=y_test)
#
#     d_test_split = []
#     t_test_split = []
#     for d_tuple in x_test:
#         d_test_split.append(d_tuple[1])
#         t_test_split.append(d_tuple[0])
#
#     return dataset_split, times_split, y_train, d_test_split, t_test_split, y_test
#
#
# def gen_datasets(n1_arr, n2_arr, c_arr, datasets_folder):
#     train_d_smalls = []
#     train_t_smalls = []
#     train_l_smalls = []
#     test_d_smalls = []
#     test_t_smalls = []
#     test_l_smalls = []
#
#     for c, n1, n2 in zip(c_arr, n1_arr, n2_arr):
#         D_s1, t_s1, l_s1, D_s1_test, t_s1_test, l_s1_test = get_small_dataset(dataset, times, labels,
#                                                                               sort_class_count_index, lclasses,
#                                                                               m, n1, n2, c)
#         train_d_smalls.append(D_s1)
#         train_t_smalls.append(t_s1)
#         train_l_smalls.append(l_s1)
#         test_d_smalls.append(D_s1_test)
#         test_t_smalls.append(t_s1_test)
#         test_l_smalls.append(l_s1_test)
#
#         # save to disk
#         np.save(datasets_folder + "train_d_n{}_c{}.npy".format(n1, c), D_s1)
#         np.save(datasets_folder + "train_t_n{}_c{}.npy".format(n1, c), t_s1)
#         np.save(datasets_folder + "train_l_n{}_c{}.npy".format(n1, c), l_s1)
#         np.save(datasets_folder + "test_d_n{}_c{}.npy".format(n2, c), D_s1_test)
#         np.save(datasets_folder + "test_t_n{}_c{}.npy".format(n2, c), t_s1_test)
#         np.save(datasets_folder + "test_l_n{}_c{}.npy".format(n2, c), l_s1_test)
#
#         print(":::GEN train dataset of size:", n1, "and test dataset of size", n2, "with", c, "classes")
#
#     return train_d_smalls, train_t_smalls, train_l_smalls, test_d_smalls, test_t_smalls, test_l_smalls
