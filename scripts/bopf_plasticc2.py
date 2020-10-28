import sys
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import multiprocessing as mp
from collections import defaultdict
from src.bopf.bopf import BagOfPatternFeature
from src.bopf.classifier import classify, classify2
from src.utils import sort_trim_arr
import numpy as np
import time


def worker_main_test_plasticc_p1(n1, n2, c, wd_arr, wl_arr, out_q):
    try:
        print("start worker")
        wd_num = len(wd_arr)
        wl_num = len(wl_arr)

        bopf = BagOfPatternFeature(special_character=True)
        path = os.path.join(main_path, "data", "plasticc_subsets", "scenario1_ratio_2-8/")
        # path = os.path.join(main_path, "data", "plasticc_sub_dataset/")
        bopf.load_dataset(path, fmt="npy", set_type="train", n1=n1, c=c)
        bopf.cumsum()

        bopf_t = BagOfPatternFeature(special_character=True)
        bopf_t.load_dataset(path, fmt="npy", set_type="test", n1=n2, c=c)
        bopf_t.cumsum()

        output_dict = defaultdict(list)

        for i in range(wd_num):
            wd = wd_arr[i]
            for j in range(wl_num):
                wl = wl_arr[j]
                print(wd, wl)
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
    except Exception as e:
        print("worker failed")
        print(e)
    finally:
        print("done")


def main_test_plasticc_p1_multiprocess(n1, n2, c, wd_arr, wl_arr, n_process):
    m = mp.Manager()
    result_queue = m.Queue()

    N = int(round((1 + len(wl_arr)) // n_process))
    jobs = []
    wl_sub_arr_tuples = []
    for k in range(n_process):
        i = k * N
        j = (k + 1) * N
        if j > len(wl_arr):
            j = len(wl_arr)
        wl_sub_arr = wl_arr[i:j]
        wl_sub_arr_tuples.append((i, j))
        jobs.append(mp.Process(target=worker_main_test_plasticc_p1,
                               args=(n1, n2, c, wd_arr, wl_sub_arr, result_queue)))
        jobs[-1].start()

    for p in jobs:
        p.join()

    output_dict = defaultdict(list)
    num_res = result_queue.qsize()
    while num_res > 0:
        wl_min, wl_max, out_dict_worker = result_queue.get()
        for k, v in out_dict_worker.items():
            output_dict[k].extend(v)
        num_res -= 1

    bopf = BagOfPatternFeature(special_character=True)
    path = os.path.join(main_path, "data", "plasticc_subsets", "scenario1_ratio_2-8/")
        # path = os.path.join(main_path, "data", "plasticc_sub_dataset/")
    bopf.load_dataset(path, fmt="npy", set_type="train", n1=n1, c=c)
    bopf.cumsum()
    bopf.bop(4, 0.9, verbose=False)
    bopf.adjust_label_set()

    bopf_t = BagOfPatternFeature(special_character=True)
    bopf_t.load_dataset(path, fmt="npy", set_type="test", n1=n2, c=c)
    bopf_t.cumsum()

    return bopf, bopf_t, output_dict


def main_test_plasticc_p2(bopf, bopf_t, output_dict, top_n):
    index1 = np.argsort(output_dict["bop_cv_acc"])[::-1]
    index2 = np.argsort(output_dict["bop_cv_acc2"])[::-1]

    best_centroid = -1
    best_tf_idf = -1
    rbest_centroid = -np.inf
    rbest_tf_idf = -np.inf
    for i in range(top_n):
        print(i, end="\r")
        s_index = index1[i]
        wd = output_dict["bop_wd"][s_index]
        wl = output_dict["bop_wl"][s_index]
        bopf_t.bop(wd, wl, verbose=False)
        test_bop_sort = sort_trim_arr(bopf_t.train_bop, output_dict["bop_feature_index"][s_index],
                                      bopf_t.m, output_dict["bop_fea_num"][s_index])
        predicted_label = classify(test_bop_sort, output_dict["bop_features"][s_index], bopf.tlabel,
                                   bopf_t.m, bopf.c, output_dict["bop_fea_num"][s_index])
        real_label = np.array(bopf_t.labels)
        count = 0
        for j in range(len(real_label)):
            if predicted_label[j] == real_label[j]:
                count += 1
        acc = count / len(real_label)
        if acc > rbest_centroid:
            rbest_centroid = acc
            best_centroid = i
    s_index1 = index1[best_centroid]
    print("classify with best centroid and wd:", output_dict["bop_wd"][s_index1],
          ", wl:", output_dict["bop_wl"][s_index1],
          "-> cv_acc:", round(output_dict["bop_cv_acc"][s_index1], 3),
          ", acc:", round(rbest_centroid, 3))

    #     else:
    # classify using tf-idf
    # classify using centroid
    for i in range(top_n):
        print(i, end="\r")
        s_index = index2[i]
        wd = output_dict["bop_wd"][s_index]
        wl = output_dict["bop_wl"][s_index]
        bopf_t.bop(wd, wl, verbose=False)
        test_bop_sort = sort_trim_arr(bopf_t.train_bop, output_dict["bop_feature_index2"][s_index],
                                      bopf_t.m, output_dict["bop_fea_num2"][s_index])
        predicted_label = classify2(test_bop_sort, output_dict["bop_features2"][s_index], bopf.tlabel,
                                    bopf_t.m, bopf.c, output_dict["bop_fea_num2"][s_index])
        real_label = np.array(bopf_t.labels)
        count = 0
        for j in range(len(real_label)):
            if predicted_label[j] == real_label[j]:
                count += 1
        acc = count / len(real_label)
        if acc > rbest_tf_idf:
            rbest_tf_idf = acc
            best_tf_idf = i

    s_index2 = index2[best_tf_idf]
    print("classify with best tf-idf and wd:", output_dict["bop_wd"][s_index2],
          ", wl:", output_dict["bop_wl"][s_index2],
          "-> cv_acc:", round(output_dict["bop_cv_acc2"][s_index2], 3),
          ", acc:", round(rbest_tf_idf, 3))

    rbest_centroid = round(rbest_centroid, 3)
    rbest_tf_idf = round(rbest_tf_idf, 3)

    if rbest_centroid > rbest_tf_idf:
        return rbest_centroid, output_dict["bop_wd"][s_index1], output_dict["bop_wl"][s_index1]
    else:
        return rbest_tf_idf, output_dict["bop_wd"][s_index2], output_dict["bop_wl"][s_index2]


if __name__ ==  '__main__':
    wd_arr = [3, 4, 5, 6, 7]
    step = 0.025
    wl_arr = np.round((np.arange(int(1/step))+1)*step, 3)
    n_process = 6
    top_n = 50

    n1_arr = [500, 1000, 2000, 4000]
    n2_arr = [125, 250, 500, 1000]
    c = 6
    for n1, n2 in zip(n1_arr, n2_arr):
        f = open((main_path + "/data/bop_plasticc_log.txt").replace("\\", "/"), "a")
        try:
            s = "========INITIALIZE=========="
            f.write(s + '\n')
            s = "train set size: %d" % n1
            print(s)
            f.write(s + '\n')
            s = "test set size: %d" % n2
            print(s)
            f.write(s + '\n')
            s = "number of classes: %d" % c
            print(s)
            f.write(s + '\n')
            ini = time.time()
            bopf3, bopf_t3, output_dict3 = main_test_plasticc_p1_multiprocess(n1, n2, c, wd_arr, wl_arr, n_process)
            end1 = time.time()
            acc2, best_wd, best_wl = main_test_plasticc_p2(bopf3, bopf_t3, output_dict3, top_n)
            end2 = time.time()
            s = "best accuracy of tests: %f" % acc2
            print(s)
            f.write(s + '\n')
            s = "parameter finder execution time: %f" % (end1 - ini)
            print(s)
            f.write(s + '\n')
            s = "test classifier execution time: %f" % (end2 - end1)
            print(s)
            f.write(s + '\n')

            f2 = open((main_path + "/data/bop_plasticc_results.csv").replace("\\", "/"), "a")
            # s = "%d,%d,%d,%f,%d,%f,%f,%f" % (n1, n2, c, acc2, best_wd, best_wl, end1-ini, end2-end1)
            s = "{},{},{},{},{},{},{},{}".format(n1, n2, c, acc2, best_wd, best_wl, end1-ini, end2-end1)
            print(s)
            f2.write(s + "\n")
            f2.close()
            s = "================================="
            print(s)
            f.write(s + '\n')
        except Exception as e:
            print(e)
            print("loop failed for n1:", n1, "n2:", n2,"c:", c)
        finally:
            f.close()