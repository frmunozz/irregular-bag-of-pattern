from collections import defaultdict
from ..utils import sort_trim_arr, load_numpy_dataset
from .classifier import classify, classify2
from .bopf import BagOfPatternFeature
import multiprocessing as mp
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def bopf_param_finder(bopf, bopf_t, wd_arr, wl_arr, window_type="fraction"):
    bopf.cumsum()
    bopf_t.cumsum()
    output_dict = defaultdict(list)
    wd_num = len(wd_arr)
    wl_num = len(wl_arr)

    for i in range(wd_num):
        wd = wd_arr[i]
        for j in range(wl_num):
            wl = wl_arr[j]
            bopf.bop(wd, wl, verbose=False, window_type=window_type)
            if len(bopf.dropped_ts) < bopf.m // 5:
                # print("passed bop and dropped_ts")
                bopf.adjust_label_set()
                # print("passed adjust_label_set")
                bopf.anova(verbose=False)
                # print("passed anova")
                bopf.anova_sort()
                # print("passed anova_sort")
                bopf.sort_trim_arr(verbose=False)
                # print("passed sort_trim_arr")
                bopf.crossVL(verbose=False)
                # print("passded crossVL")
                output_dict["bop_features"].append(bopf.crossL[:bopf.c * bopf.best_idx])
                output_dict["bop_fea_num"].append(bopf.best_idx)
                output_dict["bop_cv_acc"].append(bopf.best_score)
                output_dict["bop_feature_index"].append(bopf.sort_index[:bopf.best_idx])

                bopf.crossVL2()
                # print("passded crossVL2")
                print(wd, wl, "crossVL acc:", bopf.best_score, "crossVL2 acc:", bopf.best2_score, ", dropped ts: ", len(bopf.dropped_ts))
                output_dict["bop_features2"].append(bopf.crossL2[:bopf.c * bopf.best2_idx])
                output_dict["bop_fea_num2"].append(bopf.best2_idx)
                output_dict["bop_cv_acc2"].append(bopf.best2_score)
                output_dict["bop_feature_index2"].append(bopf.sort_index[:bopf.best2_idx])

                output_dict["bop_wd"].append(wd)
                output_dict["bop_wl"].append(wl)
            else:
                print(wd, wl, ", SEQUENCE DROPPED, len(dropped ts): ", len(bopf.dropped_ts))

    return output_dict


def bopf_param_finder_worker(train_base, test_base, wd_arr, wl_arr, path, window_type, out_q):
    try:
        print("start worker for wl range:", wl_arr[0], wl_arr[-1])
        bopf = BagOfPatternFeature(special_character=True, strategy="special2")
        bopf.load_dataset(path, fmt="npy", base=train_base)
        bopf_t = BagOfPatternFeature(special_character=True, strategy="special2")
        bopf_t.load_dataset(path, fmt="npy", base=test_base)

        bopf.cumsum()
        bopf_t.cumsum()

        output_dict = bopf_param_finder(bopf, bopf_t, wd_arr, wl_arr, window_type=window_type)
        out_q.put((wl_arr[0], wl_arr[-1], output_dict))
    except Exception as e:
        print("worker failed with error:", e)
    finally:
        print("worker done")


def bopf_param_finder_mp(path, train_base, test_base, wd_arr, wl_arr, n_process, window_type="fraction"):
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
        jobs.append(mp.Process(target=bopf_param_finder_worker,
                               args=(train_base, test_base, wd_arr, wl_sub_arr, path, window_type, result_queue)))
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

    bopf = BagOfPatternFeature(special_character=True, strategy="special2")
    bopf.load_dataset(path, fmt="npy", base=train_base)
    bopf.cumsum()
    bopf.bop(4, 0.9, verbose=False, window_type=window_type)
    bopf.adjust_label_set()

    bopf_t = BagOfPatternFeature(special_character=True, strategy="special2")
    bopf_t.load_dataset(path, fmt="npy", base=test_base)
    bopf_t.cumsum()

    return bopf, bopf_t, output_dict


def bopf_classifier_centroid(bopf, bopf_t, wd, wl, bop_feature_index, bop_features, bop_fea_num, window_type="fraction"):
    bopf_t.bop(wd, wl, verbose=False, window_type=window_type)
    print("===== Number of dropped TS: ", len(bopf_t.dropped_ts))
    test_bop_sort = sort_trim_arr(bopf_t.train_bop, bop_feature_index,
                                  bopf_t.m, bop_fea_num)
    predicted_label = classify(test_bop_sort, bop_features, bopf.tlabel,
                               bopf_t.m, bopf.c, bop_fea_num)
    for i in bopf_t.dropped_ts:
        predicted_label[i] = 111
    return predicted_label


def bopf_classifier_tf_idf(bopf, bopf_t, wd, wl, bop_feature_index, bop_features, bop_fea_num, window_type="fraction"):
    bopf_t.bop(wd, wl, verbose=False, window_type=window_type)
    print("===== Number of dropped TS: ", len(bopf_t.dropped_ts))
    test_bop_sort = sort_trim_arr(bopf_t.train_bop, bop_feature_index,
                                  bopf_t.m, bop_fea_num)
    predicted_label = classify2(test_bop_sort, bop_features, bopf.tlabel,
                                bopf_t.m, bopf.c, bop_fea_num)
    for i in bopf_t.dropped_ts:
        predicted_label[i] = 111
    return predicted_label


def bopf_classifier(bopf, bopf_t, output_dict, s_index, classifier="centroid", window_type="fraction"):
    wd = output_dict["bop_wd"][s_index]
    wl = output_dict["bop_wl"][s_index]

    if classifier == "centroid":
        feature_index = output_dict["bop_feature_index"][s_index]
        fea_num = output_dict["bop_fea_num"][s_index]
        features = output_dict["bop_features"][s_index]
        print(wd, wl, "cv_acc centroid: ", output_dict["bop_cv_acc"][s_index])
        return bopf_classifier_centroid(bopf, bopf_t, wd, wl, feature_index, features, fea_num, window_type=window_type)
    else:
        feature_index = output_dict["bop_feature_index2"][s_index]
        fea_num = output_dict["bop_fea_num2"][s_index]
        features = output_dict["bop_features2"][s_index]
        print(wd, wl, "cv_acc tf_idf: ", output_dict["bop_cv_acc2"][s_index])
        return bopf_classifier_tf_idf(bopf, bopf_t, wd, wl, feature_index, features, fea_num, window_type=window_type)


def bopf_best_classifier(bopf, bopf_t, output_dict, top_n, window_type="fraction"):
    index1 = np.argsort(output_dict["bop_cv_acc"])[::-1]
    index2 = np.argsort(output_dict["bop_cv_acc2"])[::-1]
    best_centroid = -1
    best_tf_idf = -1
    rbest_centroid = -np.inf
    rbest_tf_idf = -np.inf
    real_label = np.array(bopf_t.labels)

    top_n = min(top_n, len(index1))
    print("starting classification test")

    for i in range(top_n):
        s_index1 = index1[i]
        s_index2 = index2[i]
        pred_centroid = bopf_classifier(bopf, bopf_t, output_dict, s_index1, classifier="centroid", window_type=window_type)
        pred_tf_idf = bopf_classifier(bopf, bopf_t, output_dict, s_index2, classifier="tf_idf", window_type=window_type)

        acc_centroid = balanced_accuracy_score(real_label, pred_centroid)
        acc_tf_idf = balanced_accuracy_score(real_label, pred_tf_idf)

        print("--> acc tf_idf: ", acc_tf_idf, ", acc centroid: ", acc_centroid)

        if acc_centroid > rbest_centroid:
            rbest_centroid = acc_centroid
            best_centroid = i
        if acc_tf_idf > rbest_tf_idf:
            rbest_tf_idf = acc_tf_idf
            best_tf_idf = i

    rbest_centroid = round(rbest_centroid, 3)
    rbest_tf_idf = round(rbest_tf_idf, 3)

    s_index1 = index1[best_centroid]
    print("classify with best centroid and wd:", output_dict["bop_wd"][s_index1],
          ", wl:", output_dict["bop_wl"][s_index1],
          "-> cv_acc:", round(output_dict["bop_cv_acc"][s_index1], 3),
          ", balanced acc:", round(rbest_centroid, 3))
    s_index2 = index2[best_tf_idf]
    print("classify with best tf-idf and wd:", output_dict["bop_wd"][s_index2],
          ", wl:", output_dict["bop_wl"][s_index2],
          "-> cv_acc:", round(output_dict["bop_cv_acc2"][s_index2], 3),
          ", balanced acc:", round(rbest_tf_idf, 3))

    if rbest_centroid > rbest_tf_idf:
        pred_labels = bopf_classifier(bopf, bopf_t, output_dict, s_index1, classifier="centroid", window_type=window_type)
        return rbest_centroid, s_index1, pred_labels, real_label, output_dict, "centroid"
    else:
        pred_labels = bopf_classifier(bopf, bopf_t, output_dict, s_index2, classifier="tf_idf", window_type=window_type)
        return rbest_tf_idf, s_index2, pred_labels, real_label, output_dict, "tf_idf"