import os
import sys
import numpy as np
import pandas as pd


main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)

from src.preprocesing import single_band_dataset, gen_dataset, gen_dataset_from_h5
from src.Adeprecated.methodA import transform_to_bop, reduce_cv_bop
from sklearn.model_selection import train_test_split
from scipy import sparse
import avocado


if __name__ == "__main__":
    # df = pd.read_csv(os.path.join(main_path, "data", "training_set.csv.zip"))
    # df_meta = pd.read_csv(os.path.join(main_path, "data", "training_set_metadata.csv"))

    n_process = 8
    multiprocessing = True
    min_win = 5  # days
    max_win = 850  # days
    n_win = 40
    wl_arr = [1, 2, 3, 4]
    win_arr = np.logspace(np.log10(min_win), np.log10(max_win), n_win)
    params = {
        "special_character": True,
        "feature": "trend_value",
        "numerosity_reduction": True,
    }

    print("Generating dataset...")
    # res, labels = gen_dataset(df, df_meta)
    res, labels = gen_dataset_from_h5("plasticc_balanced_combined_classes_small_wfd")
    # res2 = single_band_dataset(res, band)

    # print("splitting dataset...")
    # train_res, test_res, train_labels, test_labels = train_test_split(res, labels,
    #                                                                   test_size=0.1,
    #                                                                   stratify=labels,
    #                                                                   random_state=42)

    tuples = []
    wl_arr_extend = []
    win_arr_extend = []
    for wl in wl_arr:
        for win in win_arr:
            tuples.append([win, wl])
            wl_arr_extend.append(wl)
            win_arr_extend.append(win)

    test_res = res
    print("size of dataset to transform:", test_res.size)
    print("running transformer using {} process".format(n_process))
    repr_arr, failed_arr = transform_to_bop(test_res, tuples, n_process=n_process,
                                            multiprocessing=multiprocessing, **params)
    repr_path = os.path.join(main_path, "data", "bop_sparse_repr")
    for i, m in enumerate(repr_arr):
        sparse.save_npz(os.path.join(repr_path, "pair_%d_repr.npz" % i), m.vector)
    dict_to_pd = {"window": win_arr_extend, "word_length": wl_arr_extend,
                  "failed_count": failed_arr}
    df_out = pd.DataFrame(dict_to_pd)
    df_out.to_csv(os.path.join(repr_path, "metadata.csv"),
                  header=True, index=True)


