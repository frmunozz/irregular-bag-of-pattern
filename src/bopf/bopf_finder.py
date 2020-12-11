from collections import defaultdict
from ..utils import sort_trim_arr, load_numpy_dataset
from .classifier import classify, classify2
from .bopf import BagOfPatternFeature
import multiprocessing as mp
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import queue
import pandas as pd


def bopf_pipeline(bopf, wd, wl, output_dict, window_type="fraction"):
    bopf.bop(wd, wl, verbose=False, window_type=window_type)
    if len(bopf.dropped_ts) <=  bopf.m // 10:

        bopf.adjust_label_set()
        bopf.anova(verbose=False)
        bopf.anova_sort()
        bopf.sort_trim_arr(verbose=False)
        bopf.crossVL(verbose=False)
        bopf.crossVL2()

        output_dict["bop_features"].append(bopf.crossL[:bopf.c * bopf.best_idx])
        output_dict["bop_fea_num"].append(bopf.best_idx)
        output_dict["bop_cv_acc"].append(bopf.best_score)
        output_dict["bop_feature_index"].append(bopf.sort_index[:bopf.best_idx])

        output_dict["bop_features2"].append(bopf.crossL2[:bopf.c * bopf.best2_idx])
        output_dict["bop_fea_num2"].append(bopf.best2_idx)
        output_dict["bop_cv_acc2"].append(bopf.best2_score)
        output_dict["bop_feature_index2"].append(bopf.sort_index[:bopf.best2_idx])

        output_dict["bop_wd"].append(wd)
        output_dict["bop_wl"].append(wl)
        output_dict["bop_dropped_ts"].append(len(bopf.dropped_ts))
        print(mp.current_process().name, wd, wl, "crossVL acc:", round(bopf.best_score, 3), "crossVL2 acc:", round(bopf.best2_score, 3), ", dropped TS: ", len(bopf.dropped_ts))
    else:
        print(mp.current_process().name, wd, wl, ", SEQUENCE DROPPED, len(dropped ts): ", len(bopf.dropped_ts))

    return output_dict



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
                output_dict["bop_dropped_ts"].append(len(bopf.dropped_ts))
            else:
                print(wd, wl, ", SEQUENCE DROPPED, len(dropped ts): ", len(bopf.dropped_ts))

    return output_dict


def bopf_param_finder_worker(combinations_to_try, train_base, test_base, path, window_type, strategy, lock, out_q):

    try: 
        print("start worker '%s'" % mp.current_process().name)
        bopf = BagOfPatternFeature(special_character=True, strategy=strategy)
        bopf.load_dataset(path, fmt="npy", base=train_base)

        bopf.cumsum()

        output_dict = defaultdict(list)

        while True:
            try:
                lock.acquire()
                wd, wl = combinations_to_try.get_nowait()
            except queue.Empty:
                lock.release()
                break
            else:
                lock.release()
                output_dict = bopf_pipeline(bopf, wd, wl, output_dict, window_type=window_type)
        out_q.put((bopf, output_dict))
    except Exception as e:
        print("Worker failed with error:", e)
    finally:
        print("worker '%s' DONE" % mp.current_process().name)


def bopf_param_finder_mp(path, train_base, test_base, wd_arr, wl_arr, n_process, out_path, window_type="fraction", strategy="special2"):

    if n_process == "default":
        n_process = mp.cpu_count()

    m = mp.Manager()
    result_queue = m.Queue()

    wd_num = len(wd_arr)
    wl_num = len(wl_arr)

    n_combinations = wd_num * wl_num
    combinations_to_try = mp.Queue()

    for i in range(wd_num):
        for j in range(wl_num):
            combinations_to_try.put((wd_arr[wd_num - (i + 1)], wl_arr[j]))

    lock = mp.Lock()

    jobs = []
    for w in range(n_process):
        p = mp.Process(target=bopf_param_finder_worker, args=(combinations_to_try, train_base, test_base, path, window_type, strategy, lock, result_queue))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    output_dict = defaultdict(list)
    num_res = result_queue.qsize()
    while num_res > 0:
        bopf, out_dict_worker = result_queue.get()
        for k, v in out_dict_worker.items():
            output_dict[k].extend(v)
        num_res -= 1

    pd.DataFrame(output_dict).to_csv(out_path, index=False)

    bopf_t = BagOfPatternFeature(special_character=True, strategy=strategy, test=True)
    bopf_t.load_dataset(path, fmt="npy", base=test_base)
    bopf_t.cumsum()

    return bopf, bopf_t, output_dict


def bopf_classifier_centroid(bopf, bopf_t, wd, wl, bop_feature_index, bop_features, bop_fea_num, window_type="fraction"):
    bopf_t.bop(wd, wl, verbose=False, window_type=window_type)
    test_bop_sort = sort_trim_arr(bopf_t.train_bop, bop_feature_index,
                                  bopf_t.m, bop_fea_num)
    predicted_label = classify(test_bop_sort, bop_features, bopf.tlabel,
                               bopf_t.m, bopf.c, bop_fea_num)
    return predicted_label, bopf_t.valid_train_bop


def bopf_classifier_tf_idf(bopf, bopf_t, wd, wl, bop_feature_index, bop_features, bop_fea_num, window_type="fraction"):
    bopf_t.bop(wd, wl, verbose=False, window_type=window_type)
    test_bop_sort = sort_trim_arr(bopf_t.train_bop, bop_feature_index,
                                  bopf_t.m, bop_fea_num)
    predicted_label = classify2(test_bop_sort, bop_features, bopf.tlabel,
                                bopf_t.m, bopf.c, bop_fea_num)
    return predicted_label, bopf_t.valid_train_bop


def bopf_classifier(bopf, bopf_t, output_dict, s_index, classifier="centroid", window_type="fraction"):
    wd = output_dict["bop_wd"][s_index]
    wl = output_dict["bop_wl"][s_index]

    if classifier == "centroid":
        feature_index = output_dict["bop_feature_index"][s_index]
        fea_num = output_dict["bop_fea_num"][s_index]
        features = output_dict["bop_features"][s_index]
        cv_acc = output_dict["bop_cv_acc"][s_index]
        pred_label, valid_train_bop = bopf_classifier_centroid(bopf, bopf_t, wd, wl, feature_index, features, fea_num, window_type=window_type)
    else:
        feature_index = output_dict["bop_feature_index2"][s_index]
        fea_num = output_dict["bop_fea_num2"][s_index]
        features = output_dict["bop_features2"][s_index]
        cv_acc = output_dict["bop_cv_acc2"][s_index]
        pred_label, valid_train_bop =  bopf_classifier_tf_idf(bopf, bopf_t, wd, wl, feature_index, features, fea_num, window_type=window_type)

    train_dropped_ts = output_dict["bop_dropped_ts"][s_index]

    info_to_print = "wd: " + str(wd) + ", wl: " + str(wl) + ", method: " + classifier
    info_to_print += ", train_dropped_ts: " + str(train_dropped_ts) + ", test_dropped_ts: " + str(len(valid_train_bop) - valid_train_bop.sum())
    info_to_print += ", cv_acc: " + str(round(cv_acc, 3))
    return pred_label, valid_train_bop, info_to_print


def check_real_pred(real_label, pred_label, valid_train_bop, drop=True):
    if drop:
        real = []
        pred = []

        for j in range(len(valid_train_bop)):
            if valid_train_bop[j]:
                real.append(real_label[j])
                pred.append(pred_label[j])

        return real, pred
    else:
        return real_label, pred_label


def bopf_best_classifier(bopf, bopf_t, output_dict, top_n, out_centroid_dict_csv, out_tf_idf_dict_csv, window_type="fraction", drop=False):
    index1 = np.argsort(output_dict["bop_cv_acc"])[::-1]
    index2 = np.argsort(output_dict["bop_cv_acc2"])[::-1]
    best_centroid = -1
    best_tf_idf = -1
    rbest_centroid = -np.inf
    rbest_tf_idf = -np.inf
    real_label = np.array(bopf_t.labels)

    if "bop_dropped_ts" not in output_dict:
        output_dict["bop_dropped_ts"] = np.zeros(len(output_dict["bop_wd"]))

    top_n = min(top_n, len(index1))
    print("starting classification test")

    out_centroid_dict = defaultdict(list)
    out_tf_idf_dict = defaultdict(list)

    for i in range(top_n):
        s_index1 = index1[i]
        s_index2 = index2[i]

        out_centroid_dict["bop_wd"].append(output_dict["bop_wd"][s_index1])
        out_centroid_dict["bop_wl"].append(output_dict["bop_wl"][s_index1])
        out_centroid_dict["cv_acc"].append(output_dict["bop_cv_acc"][s_index1])

        out_tf_idf_dict["bop_wd"].append(output_dict["bop_wd"][s_index2])
        out_tf_idf_dict["bop_wl"].append(output_dict["bop_wl"][s_index2])
        out_tf_idf_dict["cv_acc"].append(output_dict["bop_cv_acc2"][s_index2])

        pred_label, valid_train_bop, info_to_print = bopf_classifier(bopf, bopf_t, output_dict, s_index1, classifier="centroid", window_type=window_type)
        
        real_centroid, pred_centroid = check_real_pred(real_label, pred_label, valid_train_bop, drop=drop)

        acc_balanced_centroid = balanced_accuracy_score(real_centroid, pred_centroid)
        acc_centroid = accuracy_score(real_centroid, pred_centroid)
        out_centroid_dict["acc"].append(acc_centroid)
        out_centroid_dict["balanced_acc"].append(acc_balanced_centroid)


        info_to_print += ", acc: " + str(round(acc_centroid, 3)) + ", acc balanced: " + str(round(acc_balanced_centroid, 3))
        print(info_to_print)

        pred_label, valid_train_bop, info_to_print = bopf_classifier(bopf, bopf_t, output_dict, s_index2, classifier="tf_idf", window_type=window_type)


        real_tf_idf, pred_tf_idf = check_real_pred(real_label, pred_label, valid_train_bop, drop=drop)


        acc_balanced_tf_idf = balanced_accuracy_score(real_tf_idf, pred_tf_idf)
        acc_tf_idf = accuracy_score(real_tf_idf, pred_tf_idf)
        out_tf_idf_dict["acc"].append(acc_tf_idf)
        out_tf_idf_dict["balanced_acc"].append(acc_balanced_tf_idf)

        info_to_print += ", acc: " + str(round(acc_tf_idf, 3)) + ", acc balanced: " + str(round(acc_balanced_tf_idf, 3))
        print(info_to_print)

        if acc_balanced_centroid > rbest_centroid:
            rbest_centroid = acc_balanced_centroid
            best_centroid = i
        if acc_balanced_tf_idf > rbest_tf_idf:
            rbest_tf_idf = acc_balanced_tf_idf
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

    pd.DataFrame(out_centroid_dict).to_csv(out_centroid_dict_csv, index=False)
    pd.DataFrame(out_tf_idf_dict).to_csv(out_tf_idf_dict_csv, index=False)

    if rbest_centroid > rbest_tf_idf:
        pred_label, valid_train_bop, info = bopf_classifier(bopf, bopf_t, output_dict, s_index1, classifier="centroid", window_type=window_type)
        real, pred = check_real_pred(real_label, pred_label, valid_train_bop, drop=drop)

        info += ", acc balanced: " + str(rbest_centroid)
        return rbest_centroid, s_index1, pred, real, output_dict, "centroid", info
    else:
        pred_label, valid_train_bop, info = bopf_classifier(bopf, bopf_t, output_dict, s_index2, classifier="tf_idf", window_type=window_type)
        real, pred = check_real_pred(real_label, pred_label, valid_train_bop, drop=drop)

        info += ", acc balanced: " + str(rbest_tf_idf)

        return rbest_tf_idf, s_index2, pred, real, output_dict, "tf_idf", info