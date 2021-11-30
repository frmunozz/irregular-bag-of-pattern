import os
import sys
import numpy as np
import pandas as pd


main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)

from src.preprocesing import gen_dataset_from_h5
from src.Adeprecated.methodA import reduce_cv_bop
from src.Adeprecated.methodA import BOPSparseRepresentation
from scipy import sparse
from tqdm import tqdm
from collections import defaultdict
import time

merged_labels_to_num = {
    "Single microlens": 1,
    "TDE": 2,
    "Short period VS": 3,
    "SN": 4,
    "M-dwarf": 5,
    "AGN": 6,
    "Unknown": 99
}

merged_labels = {
    6: "Single microlens",
    15: "TDE",
    16: "Short period VS",
    42: "SN",
    52: "SN",
    53: "Short period VS",
    62: "SN",
    64: "SN",
    65: "M-dwarf",
    67: "SN",
    88: "AGN",
    90: "SN",
    92: "Short period VS",
    95: "SN",
    99: "Unknown"
}

if __name__ == "__main__":
    data_path = os.path.join(main_path, "data", "bop_sparse_repr")
    df = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    n_process = 8
    multiprocessing = True
    success_rate_threshold = 0.95

    res, labels = gen_dataset_from_h5("plasticc_balanced_combined_classes_small_wfd")
    labels = [merged_labels_to_num[merged_labels[x]] for x in labels]

    m = len(res)
    labels = np.array(labels)
    mean_n = np.mean([x.n for x in res])
    n_components = int(2 * mean_n - 1)
    print("mean number of observations on each ts: {}, n_components to use: {}".format(mean_n, n_components))

    repr_arr = np.zeros(df.shape[0], dtype=object)
    combinations = []
    combis_idx = []
    # for i in range(df.shape[0]):
    for i in tqdm(
                range(df.shape[0]), desc="Preparation", dynamic_ncols=True
        ):
        metadata = dict(df.iloc[i])
        if metadata["word_length"] >= 4:
            continue
        combinations.append([[metadata["window"], metadata["word_length"]]])
        combis_idx.append([i])
        repr_i = BOPSparseRepresentation()
        repr_i.store_repr(sparse.load_npz(os.path.join(data_path, "pair_%d_repr.npz" % i)))
        # repr_i.as_class_centroid(labels)
        repr_i.as_tf_idf()
        # repr_i.sample_wise_norm()
        repr_arr[i] = repr_i
    print("total combis to use:", len(combinations))
    print(combinations[0][0])
    ini = time.time()
    _, acc_arr, failed_arr = reduce_cv_bop(repr_arr, labels, combinations, combis_idx, n_components)
    end = time.time()
    print("EXECUTION TIME OF FIRST ITERATION: ", end-ini)
    p_best = 40
    idxs = np.argsort(acc_arr)[::-1]
    idxs = idxs[:p_best]

    # top_p_combinations = np.array(combinations)[idxs]
    top_p_acc = np.array(acc_arr)[idxs]

    output_dict = defaultdict(list)
    for i in idxs[:5]:
        prev_combis_idx = [i]
        prev_combinations = [combinations[i]]
        best_acc = acc_arr[i]
        counter = 0
        while True:
            if len(prev_combis_idx) > 5:
                break

            combis_idx_top = []
            combinations_top = []
            for j in idxs:
                if j not in prev_combis_idx:
                    tmp = prev_combis_idx.copy()
                    tmp.append(j)
                    combis_idx_top.append(tmp)
                    tmp2 = prev_combinations.copy()
                    tmp2.append(combinations[j])
                    combinations_top.append(tmp2)
            ini = time.time()
            _, acc_arr_sub, failed_arr_sub = reduce_cv_bop(repr_arr, labels, combinations_top, combis_idx_top, n_components)
            end = time.time()
            print("execution time of iteration [%d, %d]: %.3f" % (i, counter, end-ini))
            idxs_top = np.argsort(acc_arr_sub)[::-1]
            print(acc_arr_sub[idxs_top])
            best_success_rate = (m - failed_arr[idxs_top[0]]) / m

            if success_rate_threshold <= best_success_rate and best_acc > acc_arr_sub[idxs_top[0]]:
                break

            prev_combis_idx = combis_idx_top[idxs_top[0]]
            prev_combinations = combinations_top[idxs_top[0]]
            best_acc = acc_arr_sub[idxs_top[0]]
            print("new best acc: {}, for combis: {}".format(best_acc, prev_combis_idx))
            counter += 1
        print("best combinations {} [iter {}] gives cv_acc: {}".format(prev_combis_idx, i, best_acc))
        for idx_i in prev_combis_idx:
            output_dict["id"].append(i)
            win, wl = combinations[idx_i][0]
            output_dict["window"].append(win)
            output_dict["word_length"].append(wl)
            output_dict["cv_acc"].append(best_acc)
            output_dict["n_coms"].append(n_components)

    pd.DataFrame(output_dict).to_csv(os.path.join(main_path, "data", "results_plasticc_balanced_combined_classes_small_wfd.csv"), index=False, header=True)