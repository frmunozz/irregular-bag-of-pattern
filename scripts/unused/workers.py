import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_path)
from src.bopf.bopf import BagOfPatternFeature
from collections import defaultdict


def worker_main_test_plasticc_p1(n1, n2, c, wd_arr, wl_arr, out_q):
    try:
        print("start worker")
        wd_num = len(wd_arr)
        wl_num = len(wl_arr)

        bopf = BagOfPatternFeature(special_character=True)
        path = "D:/tesis/tesis/data/plasticc_sub_dataset/"
        bopf.load_dataset(path, fmt="npy", set_type="train", n1=n1, c=c)
        bopf.cumsum()

        bopf_t = BagOfPatternFeature(special_character=True)
        bopf_t.load_dataset(path, fmt="npy", set_type="train", n1=n2, c=c)
        bopf_t.cumsum()

        output_dict = defaultdict(list)

        for i in range(wd_num):
            wd = wd_arr[i]
            for j in range(wl_num):
                wl = wl_arr[j]
                bopf.bop(wd, wl, verbose=False)
                bopf.adjust_label_set()
                bopf.anova(verbose=False)
                bopf.anova_sort()
                bopf.sort_trim_arr(verbose=False)
                bopf.crossVL(verbose=False)
                output_dict["bop_features"].append(bopf.crossL[:bopf.c * bopf.best_idx])
                output_dict["bop_fea_num"].append(bopf.best_idx)
                output_dict["bop_cv_acc"].append(bopf.best_score)
                output_dict["bop_feature_index"].append(bopf.sort_index[:bopf.best_idx])

                bopf.crossVL2()
                output_dict["bop_features2"].append(bopf.crossL2[:bopf.c * bopf.best2_idx])
                output_dict["bop_fea_num2"].append(bopf.best2_idx)
                output_dict["bop_cv_acc2"].append(bopf.best2_score)
                output_dict["bop_feature_index2"].append(bopf.sort_index[:bopf.best2_idx])

                output_dict["bop_wd"].append(wd)
                output_dict["bop_wl"].append(wl)

        out_q.put((wl_arr[0], wl_arr[-1], output_dict))
    except:
        print("worker failed")
    finally:
        print("done")
