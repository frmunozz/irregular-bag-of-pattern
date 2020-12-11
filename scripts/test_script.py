import os
import sys

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)
import numpy as np

from src.methodA.multitask import bopf_transformer_mp, bopf_transformer_separated_mp
from src.methodA.transformer import BOPFTransformer
from src.utils import read_numpy_dataset, read_file_regular_dataset, plot_confusion_matrix
from src.methodA.class_vectors import predict_by_euclidean, predict_by_cosine
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ucr_data = os.path.join(main_path, "data", "UCRArchive_2018")
    plasticc_data = os.path.join(main_path, "data", "plasticc_subsets", "ddf_wdf_split_min_detection")

    # key = "MiddlePhalanxTW"
    # key = "DistalPhalanxOutlineAgeGroup"
    # key = "FaceFour"
    # key = "Wine"
    # train_file = os.path.join(ucr_data, key, key + "_TRAIN.tsv")
    # test_file = os.path.join(ucr_data, key, key + "_TEST.tsv")
    # d_train, t_train, l_train, m = read_file_regular_dataset(train_file)
    # d_test, t_test, l_test, m = read_file_regular_dataset(test_file)

    # key = "3_min"
    # survey = "wdf"
    # base = survey + "_%s_%s.npy"
    # train_set_files = [os.path.join(plasticc_data, key, base % ("train", x)) for x in ["d", "t", "l"]]
    # test_set_files = [os.path.join(plasticc_data, key, base % ("test", x)) for x in ["d", "t", "l"]]
    # d_train, t_train, l_train, m = read_numpy_dataset(train_set_files[0], train_set_files[1], train_set_files[2])
    # d_test, t_test, l_test, m_test = read_numpy_dataset(test_set_files[0], test_set_files[1], test_set_files[2])

    key = "5_min"
    survey = "wdf"
    base = survey + "_%s.npy"
    set_files = [os.path.join(plasticc_data, key, base % x) for x in ["dataset", "times", "labels"]]
    dataset, times, labels, m = read_numpy_dataset(set_files[0], set_files[1], set_files[2])
    d_train ,d_test, t_train, t_test, l_train, l_test = train_test_split(dataset, times, labels, test_size=0.3,
                                                                         stratify=labels)
    print("train size:", len(d_train), ", test size:", len(d_test))

    special_character = True
    feature = "trend_value"
    tol = 2
    numerosity_reduction = True
    method = "centroid"

    print("Using Dataset ", key, "with", len(d_train), "time series in train set")
    mean_width = np.mean([x[-1] - x[0] for x in t_train])
    print("Mean time duraton of each time series in train set is: ", mean_width)

    wl_arr = np.array([1, 2, 3], dtype=int)

    win_arr = np.round(np.linspace(7, mean_width, 20), 2)
    # print(win_arr)
    # win_arr = [19.1, 20.97, 22.85, 24.72, 26.59, 28.46, 30.33, 32.21]
    # win_arr = win_arr[:13]
    # print(win_arr)
    tuples_to_try = []
    for wl in wl_arr:
        for win in win_arr:
            tuples_to_try.append((win, wl))


    # transf = BOPFTransformer(special_character=special_character, feature=feature,
    #                                                   tol=tol, numerosity_reduction=numerosity_reduction)
    # transf.set_full_alphabet(transf.get_alph_size())
    # # transf.logger.info("start BOPFTransformer on worker '%s'" % mp.current_process().name)
    # transf.alphabet = transf["full_alphabet"][:transf.get_alph_size()]
    # print(len(transf.alphabet))
    # transf.bop(d_train, t_train, 3, 21)
    # _ = transf.count_failed()
    # fvalues, _, _, class_count, positions = transf.anova(len(d_train), l_train, 3)
    # _, limit = transf.reduce_zeros(fvalues)
    # res, yut, vec = transf.cv_fea_num_centroid(limit, np.array(l_train), positions, class_count)
    # print(res)


    ini = time.time()
    out_dict_centroid = bopf_transformer_separated_mp((d_train, t_train, l_train), tuples_to_try,
                                   method=method, n_process=8,
                                   special_character=special_character, feature=feature,
                                                      tol=tol, numerosity_reduction=numerosity_reduction)
    end = time.time()
    print("TIME:", end-ini)


    # print(out_dict)
    if len(out_dict_centroid["bacc"]) == 0:
        print("something wrong!")
        raise ValueError()
    # import pdb
    # pdb.set_trace()
    idx_base = np.where(np.array(out_dict_centroid["failed"]) == 0)[0]
    bacc_centroid_sort_idx = np.argsort(np.array(out_dict_centroid["bacc"])[idx_base])[::-1]
    bacc_c_sort_all = np.argsort(np.array(out_dict_centroid["bacc"]))[::-1]
    idx1 = idx_base[bacc_centroid_sort_idx[0]]

    print("[{}] best bacc={} for k={} using win={}, wl={}".format(method.upper(), out_dict_centroid["bacc"][idx1],
                                                                        out_dict_centroid["fea_num"][idx1],
                                                                        out_dict_centroid["win"][idx1],
                                                                        out_dict_centroid["wl"][idx1]))

    N = 13
    M = 4
    best_pred = None
    best_bacc = -1

    for i, idx_i in enumerate(bacc_centroid_sort_idx[:M]):
        idx = idx_base[idx_i]
        print("executing for best idx: ", idx, ", n_iter:", i+1)
        current_bacc_centroid = out_dict_centroid["bacc"][idx]
        centroid_tuples = [(out_dict_centroid["win"][idx], out_dict_centroid["wl"][idx])]
        k_centroid = [out_dict_centroid["fea_num"][idx]]
        fvalues_centroid = [out_dict_centroid["fvalues"][idx]]
        yin = out_dict_centroid["yout"][idx]
        print("[{}] best bacc: {} for tuples: {}".format(method.upper(), current_bacc_centroid, centroid_tuples))
        print("intializing loop")
        counter = 2
        while True:
            if counter == 5:
                break
            tuples_to_try2 = []
            for wl, win in zip(np.array(out_dict_centroid["wl"])[bacc_c_sort_all[:N+1]],
                               np.array(out_dict_centroid["win"])[bacc_c_sort_all[:N+1]]):
                if (win, wl) not in centroid_tuples:
                    tuples_to_try2.append((win, wl))
                else:
                    print("skipping tuple: ({},{}) since it was already added".format(win, wl))

            # for pair in tuples_to_try:
            #     if pair not in centroid_tuples:
            #         tuples_to_try2.append((win, wl))

            ini = time.time()
            out_dict_c = bopf_transformer_separated_mp((d_train, t_train, l_train), tuples_to_try2,
                                                   method=method, n_process=8,
                                                   yin=yin,
                                                   special_character=special_character, feature=feature,
                                                      tol=tol, numerosity_reduction=numerosity_reduction)
            end = time.time()
            print("TIME :", end - ini)
            if len(out_dict_c["bacc"]) == 0:
                print("something wrong!")
                break

            bacc_c_sort_idx = np.argsort(out_dict_c["bacc"])[::-1]
            idx01 = bacc_c_sort_idx[0]
            if current_bacc_centroid <= out_dict_c["bacc"][idx01]:
                current_bacc_centroid = out_dict_c["bacc"][idx01]
                centroid_tuples.append((out_dict_c["win"][idx01], out_dict_c["wl"][idx01]))
                k_centroid.append(out_dict_c["fea_num"][idx01])
                fvalues_centroid.append(out_dict_c["fvalues"][idx01])
                print("[{}] best bacc: {} for tuples: {} using k:{}".format(method.upper(), current_bacc_centroid,
                                                                                  centroid_tuples,
                                                                                  out_dict_c["fea_num"][idx01]))
                counter += 1
            else:
                print("[{}] current iteration bacc: {} is not higher than previous. loop stopped".format(
                    method.upper(), out_dict_c["bacc"][idx01]))
                break

        print("[{}] best bacc: {} for tuples: {}".format(method.upper(), current_bacc_centroid, centroid_tuples))
        print("k:", k_centroid)
        print("fvalues len:", [len(x) for x in fvalues_centroid])
        print("computing TRAIN set vectors")
        ini = time.time()
        out_dict_c_vec = bopf_transformer_separated_mp((d_train, t_train, l_train), centroid_tuples,
                                                   method=method, n_process=8,
                                                   yin=None, only_vectors=True, test_fvalues=fvalues_centroid,
                                                   special_character=special_character, feature=feature,
                                                      tol=tol, numerosity_reduction=numerosity_reduction)
        end = time.time()
        print("TIME TRAIN ONLY VECTORS:", end - ini)
        sort_i = np.argsort(out_dict_c_vec["i"])
        # vecs = np.array(out_dict_c_vec["vec"])[sort_i]
        vectors = out_dict_c_vec["vec"][sort_i[0]][:, :k_centroid[0]]
        print(vectors.shape)
        if len(sort_i) > 0:
            for i in range(1, len(sort_i)):
                print(out_dict_c_vec["vec"][sort_i[i]][:,:k_centroid[i]].shape)
                vectors = np.concatenate((vectors, out_dict_c_vec["vec"][sort_i[i]][:, :k_centroid[i]]), axis=1)

        print("computing TEST set vectors")
        ini = time.time()
        out_dict_c_vec = bopf_transformer_separated_mp((d_test, t_test, l_test), centroid_tuples,
                                                       method=method, n_process=8,
                                                       yin=None, only_vectors=False, test_fvalues=fvalues_centroid,
                                                       special_character=special_character, feature=feature,
                                                      tol=tol, numerosity_reduction=numerosity_reduction)
        end = time.time()
        print("TIME TEST ONLY VECTORS:", end - ini)
        sort_i = np.argsort(out_dict_c_vec["i"])
        # vecs = np.array(out_dict_c_vec["vec"])[sort_i]
        vectors_test = out_dict_c_vec["vec"][sort_i[0]][:, :k_centroid[0]]
        print(vectors_test.shape)
        if len(sort_i) > 0:
            for i in range(1, len(sort_i)):
                print(out_dict_c_vec["vec"][sort_i[i]][:, :k_centroid[i]].shape)
                vectors_test = np.concatenate((vectors_test, out_dict_c_vec["vec"][sort_i[i]][:, :k_centroid[i]]), axis=1)

        print(vectors.shape, vectors_test.shape)
        print("evaluating method")
        classes = np.unique(l_train)
        pred_euc = predict_by_euclidean(vectors, classes, vectors_test)
        pred_cos = predict_by_cosine(vectors, classes, vectors_test)
        acc_euc = balanced_accuracy_score(l_test, pred_euc)
        acc_cos = balanced_accuracy_score(l_test, pred_cos)
        print("accuracy by euclidean: ", acc_euc)
        print("accuracy by cosine:", acc_cos)
        if method == "centroid":
            if best_bacc <= acc_euc:
                best_bacc = acc_euc
                best_pred = pred_euc
        else:
            if best_bacc <= acc_cos:
                best_bacc = acc_cos
                best_pred = pred_cos
            # best_bacc = acc
            # best_pred = pred

    conf = confusion_matrix(l_test, best_pred)
    print("final best balanced accuracy:", balanced_accuracy_score(l_test, best_pred))
    plt.figure()
    plot_confusion_matrix(conf, np.unique(l_train), normalize=False, title="Conf. Matrix [key {}, survey {}]".format(key, survey))
    plt.savefig("conf_matrix_{}_{}_{}".format(method, key, survey))




    # ini = time.time()
    # out_dict_tf_idf = bopf_transformer_separated_mp((d_train, t_train, l_train), tuples_to_try,
    #                                                 method="tf_idf", n_process=8,
    #                                                 special_character=False)
    # end = time.time()
    # print("TIME CENTROID:", end - ini)
    #
    # bacc_tf_idf_sort_idx = np.argsort(out_dict_tf_idf["bacc"][::-1])
    #
    # idx2 = bacc_tf_idf_sort_idx[0]
    #
    # print("[TFIDF] best bacc={} for k={} using win={}, wl={}".format(out_dict_tf_idf["bacc"][idx2],
    #                                                                  out_dict_tf_idf["fea_num"][idx2],
    #                                                                  out_dict_tf_idf["win"][idx2],
    #                                                                  out_dict_tf_idf["wl"][idx2]))
    #
    # current_bacc_tf_idf = out_dict_tf_idf["bacc"][idx2]
    # tf_idf_tuples = [(out_dict_tf_idf["win"][idx2], out_dict_tf_idf["wl"][idx2])]
    # k_tf_idf = [out_dict_tf_idf["fea_num_tf_idf"][idx2]]
    # yin = out_dict_centroid["yout"][idx1]
    # print("[CENTROID] best bacc: {} for tuples: {}".format(current_bacc_centroid, centroid_tuples))
    # print("intializing loop")
    # while out_dict_tf_idf["bacc"][idx2] >= current_bacc_tf_idf:
    #     pass

    # :::::::::: second iter :::::::::::;

    # N = 10
    # wl_arr2 = np.array(out_dict["wl"])[bacc_centroid_sort_idx[1:N]]
    # wl_arr2 = np.append(wl_arr2, np.array(out_dict["wl"])[bacc_tf_idf_sort_idx[1:N]])
    # win_arr2 = np.array(out_dict["win"])[bacc_centroid_sort_idx[1:N]]
    # win_arr2 = np.append(win_arr2, np.array(out_dict["win"])[bacc_tf_idf_sort_idx[1:N]])
    # y1 = out_dict["y1"][idx1]
    # y2 = out_dict["y2"][idx2]
    # prev_centroid = (out_dict["win"][idx1], out_dict["wl"][idx1])
    # prev_tf_idf = (out_dict["win"][idx2], out_dict["wl"][idx2])

    # tuples_to_try = []
    # for win_i, wl_i in zip(win_arr2, wl_arr2):
    #     if (win_i, wl_i) not in tuples_to_try:
    #         tuples_to_try.append((win_i, wl_i))
    # wl_arr = np.array([4], dtype=int)

    # win_arr = np.round(np.linspace(6, mean_width, 40), 2)
    # win_arr = [6]
    # tuples_to_try = []
    # for wl in wl_arr:
    #     for win in win_arr:
    #         tuples_to_try.append((win, wl))




    # ini = time.time()
    # print("initializing second iteration, we start from prev best iteration")
    # print("trying %d tuples" % len(tuples_to_try))
    # out_dict = bopf_transformer_mp((d_train, t_train, l_train), tuples_to_try,
    #                                n_splits=n_splits, n_process=8, y1=y1, y2=y2,
    #                                prev_centroid=prev_centroid, prev_tf_idf=prev_tf_idf,
    #                                special_character=False)
    # end = time.time()
    # print("TIME:", end - ini)
    #
    # bacc_centroid_sort_idx = np.argsort(out_dict["bacc_centroid"])[::-1]
    # bacc_tf_idf_sort_idx = np.argsort(out_dict["bacc_tf_idf"])[::-1]
    # idx1 = bacc_centroid_sort_idx[0]
    # idx2 = bacc_tf_idf_sort_idx[0]
    # print("[CENTROID] best bacc={} for k={} using win={}, wl={}".format(out_dict["bacc_centroid"][idx1],
    #                                                                     out_dict["fea_num_centroid"][idx1],
    #                                                                     out_dict["win"][idx1],
    #                                                                     out_dict["wl"][idx1]))
    # print("[TFIDF] best bacc={} for k={} using win={}, wl={}".format(out_dict["bacc_tf_idf"][idx1],
    #                                                                  out_dict["fea_num_tf_idf"][idx1],
    #                                                                  out_dict["win"][idx1],
    #                                                                  out_dict["wl"][idx1]))


    # :::::::::::::::; third iter ::::::::::::;

    # wl_arr2 = np.array(out_dict["wl"])[bacc_centroid_sort_idx[1:]]
    # wl_arr2 = np.append(wl_arr2, np.array(out_dict["wl"])[bacc_tf_idf_sort_idx[1:]])
    # win_arr2 = np.array(out_dict["win"])[bacc_centroid_sort_idx[1:]]
    # win_arr2 = np.append(win_arr2, np.array(out_dict["win"])[bacc_tf_idf_sort_idx[1:]])
    # y1 = out_dict["y1"][idx1]
    # y2 = out_dict["y2"][idx2]
    # prev_centroid = (out_dict["win"][idx1], out_dict["wl"][idx1])
    # prev_tf_idf = (out_dict["win"][idx2], out_dict["wl"][idx2])

    # tuples_to_try = []
    # for win_i, wl_i in zip(win_arr2, wl_arr2):
    #     if (win_i, wl_i) not in tuples_to_try:
    #         tuples_to_try.append((win_i, wl_i))

    # wl_arr = np.array([2, 3, 4, 5, 6], dtype=int)
    #
    # win_arr = np.round(np.linspace(6, mean_width, 40), 2)
    # # win_arr = [6]
    # tuples_to_try = []
    # for wl in wl_arr:
    #     for win in win_arr:
    #         tuples_to_try.append((win, wl))

    # ini = time.time()
    # print("initializing third iteration, we start from prev best iteration")
    # print("trying %d tuples" % len(tuples_to_try))
    # out_dict = bopf_transformer_mp((d_train, t_train, l_train), tuples_to_try,
    #                                n_splits=n_splits, n_process=8, y1=y1, y2=y2,
    #                                prev_centroid=prev_centroid, prev_tf_idf=prev_tf_idf,
    #                                special_character=False)
    # end = time.time()
    # print("TIME:", end - ini)
    #
    # bacc_centroid_sort_idx = np.argsort(out_dict["bacc_centroid"])[::-1]
    # bacc_tf_idf_sort_idx = np.argsort(out_dict["bacc_tf_idf"])[::-1]
    # idx1 = bacc_centroid_sort_idx[0]
    # idx2 = bacc_tf_idf_sort_idx[0]
    # print("[CENTROID] best bacc={} for k={} using win={}, wl={}".format(out_dict["bacc_centroid"][idx1],
    #                                                                     out_dict["fea_num_centroid"][idx1],
    #                                                                     out_dict["win"][idx1],
    #                                                                     out_dict["wl"][idx1]))
    # print("[TFIDF] best bacc={} for k={} using win={}, wl={}".format(out_dict["bacc_tf_idf"][idx1],
    #                                                                  out_dict["fea_num_tf_idf"][idx1],
    #                                                                  out_dict["win"][idx1],
    #                                                                  out_dict["wl"][idx1]))