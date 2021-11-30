# -*- coding: utf-8 -*-
import avocado
import os
import time


_SYMBOLS = {
    "mean": "Me",
    "std": "St",
    "trend": "Tr",
    "min_max": "Mm",
    "min": "Mn",
    "max": "Mx",
    "var": "Va",
    "count": "Co"
}

_REVERSE_SYMBOLS = {
    "Me": "mean",
    "St": "std",
    "Tr": "trend",
    "Mm": "min_max",
    "Mn": "min",
    "Mx": "max",
    "Va": "var",
    "Co": "count"
}


def quantity_code_extend(q_code):
    if q_code[0] == "(":
        q_code = q_code[1:]
    if q_code[-1] == ")":
        q_code = q_code[:-1]

    f_arr = []
    for q in q_code.split("-"):
        if len(q) > 2:
            # we have more than 1 quantity with early-fusion
            i = 0
            N = len(q)
            ff_arr = []
            while i < N:
                q_i = q[i:i + 2]
                ff_arr.append(_REVERSE_SYMBOLS[q_i])
                i += 2
            f_arr.append(ff_arr)
        else:
            f_arr.append([_REVERSE_SYMBOLS[q]])

    return f_arr


def check_file_path(path, name):
    file_path = os.path.join(path, name)
    if os.path.exists(file_path):
        name_arr = name.split()
        if len(name_arr) == 1:
            new_name = "%s_%s" % (name_arr[0], time.strftime("%Y%m%d-%H%M%S"))
        elif len(name_arr) >= 2:
            new_name = "%s_%s.%s" % (".".join(name_arr[:-1]), time.strftime("%Y%m%d-%H%M%S"), name_arr[-1])
        else:
            raise ValueError("invalid name '%s'" % name)
        print("WARNING file '%s' already exists, generating new file '%s'" % (name, new_name))
        file_path = os.path.join(path, new_name)
    return file_path


def write_compact_features(name, data, chunk=None, num_chunks=None,
                           settings_dir="", check_file=True):
    classifier_directory = avocado.settings[settings_dir]
    features_directory = os.path.join(classifier_directory, "features")
    if not os.path.exists(features_directory):
        os.mkdir(features_directory)
    if check_file:
        file_path = check_file_path(features_directory, name)
    else:
        file_path = os.path.join(features_directory, name)

    avocado.write_dataframe(file_path, data, "features", chunk=chunk, num_chunks=num_chunks)
