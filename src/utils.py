import pandas as pd
import numpy as np


def load_pandas(path,  **kwargs):
    data_filename = kwargs.pop("data_filename")
    meta_filename = kwargs.pop("meta_filename")
    df = pd.read_csv(path + data_filename)
    df_metadata = pd.read_csv(path + meta_filename)

    passband_id = 3

    df = df[df["passband"] == passband_id]
    df = df.sort_values(by=["object_id", "mjd"])
    df_metadata = df_metadata.sort_values(by=["object_id"])
    df = df.groupby("object_id")
    fluxes = df['flux'].apply(list)
    times = df['mjd'].apply(list)
    ids = df.groups.keys()
    dataset = [np.array(fluxes.loc[i]) for i in ids]
    times_arr = []
    for i in ids:
        times_i = np.array(times.loc[i])
        times_i = times_i - times_i[0]
        times_arr.append(times_i)

    labels = df_metadata["target"].to_numpy()

    return dataset, times_arr, labels, len(dataset)


def adjust_labels(labels):
    m = len(labels)
    classes = np.sort(np.unique(labels))
    classes_count = np.zeros(len(classes), dtype=int)
    label_index = np.zeros(m)
    for i, l in enumerate(labels):
        position = np.where(classes == l)[0][0]
        classes_count[position] += 1
        label_index[i] = position

    count_sort_best_index = np.argsort(classes_count)[::-1]

    return classes_count, classes, label_index, count_sort_best_index
