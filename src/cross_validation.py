import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from collections import defaultdict
import time
from scipy import sparse
import pandas as pd
import glob

"""
the folowing function definition are based on:

Single-Resolution Multi-quantity Multi-variate BOPF (SMM-BOPF)
Multi-Resolution Multi-quantity  Multi-variate BOPF (MMM-BOPF)

"""

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]


def cv_score(x, labels, classes, _pipeline, message="", cv=5, n_jobs=8):
    ini = time.time()
    dropped, max_dropped = _pipeline.check_dropped(x, labels)
    if dropped > max_dropped:
        print("[{}] dropped because {} > {}".format(message, dropped, max_dropped))
        return [-1], None, dropped, [-1, -1], 0

    n_variables = len(_BANDS)
    n_observations, m_temp = x.shape
    n_features = m_temp // n_variables
    # print(":::::::: initial matrix shape:", x.shape,
    #       "(n_observations: %d, n_features: %d, n_variables: %d)" % (n_observations,
    #                                                                  n_features,
    #                                                                  n_variables))
    sklearn_pipeline = _pipeline.get_compact_pipeline(n_variables, n_features, classes)

    print("[%s]: DOING CROSS VALIDATION..." % message, end="\r")
    ini = time.time()
    scores = cross_val_score(sklearn_pipeline, x, labels,
                             scoring="balanced_accuracy", cv=cv, n_jobs=n_jobs, verbose=0)
    end = time.time()
    print("[%s]: %.3f += %.3f (time: %.3f sec)" % (message, float(np.mean(scores)),
                                                   float(np.std(scores)), end - ini))

    return scores, sklearn_pipeline, dropped, [-1, -1], end - ini


def cv_ssm_bopf(data, labels, wins, wls, _pipeline, cv=5, n_jobs=8,
                drop_zero_variance=True, outfile=None, C="LSA"):
    classes = np.unique(labels)
    q_code = _pipeline.Q[0]
    data_mr_repr = defaultdict(lambda: defaultdict(object))
    cv_results = defaultdict(lambda: defaultdict(object))
    result_lists = defaultdict(list)
    best_data = None
    best_acc = -1
    print("LOADING DATA FROM QUANTITY SEARCH... ", end="")
    iniq = time.time()
    for wl in wls:
        for win in wins:
            message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
            # try:
            if True:
                data_repr_i_full, elapse = _pipeline.ssm_bopf(data, win, wl, q_code)
                message += "{BOPF time: %.3f Secs}" % elapse
                if drop_zero_variance:
                    try:
                        drop_zero_variance = VarianceThreshold()
                        data_repr_i = drop_zero_variance.fit_transform(data_repr_i_full)
                    except:
                        data_repr_i = data_repr_i_full
                else:
                    data_repr_i = data_repr_i_full
                # print(":::::::: zero variance reduces features: ",
                #       data_repr_i_full.shape, " -> ", data_repr_i.shape)
                data_mr_repr[win][wl] = data_repr_i
                cv_results_i = cv_score(data_repr_i, labels, classes, _pipeline, message=message, cv=cv, n_jobs=n_jobs)
                cv_results[win][wl] = cv_results_i
                result_lists["win"].append(win)
                result_lists["wl"].append(wl)

                if cv_results_i[1] is not None:
                    result_lists["score"].append(np.mean(cv_results_i[0]))
                else:
                    result_lists["score"].append(-1)
                acc_i = float(np.mean(cv_results_i[0]))
                if acc_i > best_acc:
                    best_acc = acc_i
                    best_data = (cv_results_i, win, wl)
                if outfile is not None:
                    cv_mean = float(np.mean(cv_results_i[0]))
                    cv_std = float(np.std(cv_results_i[0]))
                    line = "%s,%s,%.3f,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f" % (
                        C, q_code, win, wl, _pipeline.doc_kw["alphabet_size"],
                        cv_results_i[2], str(cv_mean > -1), cv_mean, cv_std,
                        elapse, cv_results_i[4])
                    f = open(outfile, "a+")
                    f.write(line + "\n")
                    f.close()
            # except Exception as e:
            #     print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))
    endq = time.time()
    print("DONE (%.3f secs)" % endq - iniq)
    return data_mr_repr, cv_results, result_lists, best_acc, best_data


def cv_smm_bopf(data, labels, wins, wls, _pipeline, cv=5, n_jobs=8,
                drop_zero_variance=True, outfile=None, C="-", data_folder=None):
    """
    compute cross validation score for

    Single-Resolution Multi-quantity Multi-variate BOPF (SMM-BOPF)

    using different combinations of (window width, word length)
    """
    classes = np.unique(labels)
    q_code = _pipeline.quantities_code()
    data_mr_repr = defaultdict(lambda: defaultdict(object))
    cv_results = defaultdict(lambda: defaultdict(object))
    result_lists = defaultdict(list)
    best_data = None
    best_acc = -1

    for wl in wls:
        for win in wins:
            try:
                if data_folder is None:
                    data_repr_i_full, elapse = _pipeline.smm_bopf(data, win, wl)
                    if drop_zero_variance:
                        drop_zero_variance = VarianceThreshold()
                        data_repr_i = drop_zero_variance.fit_transform(data_repr_i_full)
                    else:
                        data_repr_i = data_repr_i_full
                else:
                    _pipeline.doc_kw["alphabet_size"] = np.array([_pipeline.alpha])
                    quantities_keys = _pipeline.quantities_code()[1:-1].split("-")
                    data_keys = []
                    for q_key in quantities_keys:
                        data_key = sparse.load_npz(os.path.join(data_folder, q_key + "_%d_%.3f.npz" % (wl, win)))
                        data_keys.append(data_key)
                    data_repr_i = sparse.hstack(data_keys, format="csr")
                    elapse = -1.0
                # print(":::::::: zero variance reduces features: ",
                #       data_repr_i_full.shape, " -> ", data_repr_i.shape)
                data_mr_repr[win][wl] = data_repr_i
                message = "[win: %.3f, wl: %d, q: %s, n_fea: %d, bopf_time: %.3f]" % (
                    win, wl, q_code, data_repr_i.shape[1], elapse)
                cv_results_i = cv_score(data_repr_i, labels, classes, _pipeline, message=message, cv=cv, n_jobs=n_jobs)
                cv_results[win][wl] = cv_results_i
                result_lists["win"].append(win)
                result_lists["wl"].append(wl)

                if cv_results_i[1] is not None:
                    result_lists["score"].append(np.mean(cv_results_i[0]))
                else:
                    result_lists["score"].append(-1)
                acc_i = float(np.mean(cv_results_i[0]))
                if acc_i > best_acc:
                    best_acc = acc_i
                    best_data = (cv_results_i, win, wl)
                if outfile is not None:
                    cv_mean = float(np.mean(cv_results_i[0]))
                    cv_std = float(np.std(cv_results_i[0]))
                    line = "%s,%s,%.3f,%d,%d,%d,%s,%s,%.3f,%.3f,%.3f,%.3f" % (
                        C, q_code, win, wl, _pipeline.doc_kw["alphabet_size"][0],
                        cv_results_i[2], str(data_repr_i.shape[1]), str(cv_mean > -1),
                        cv_mean, cv_std, elapse, cv_results_i[4])
                    f = open(outfile, "a+")
                    f.write(line + "\n")
                    f.close()

            except Exception as e:
                print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))

    return data_mr_repr, cv_results, result_lists, best_acc, best_data


def cv_mmm_bopf(data, labels, _pipeline, cv=5, wins=None, wls=None,
                resolution_max=4, top_k=4, out_path=None,
                out_base_bopf_path=None, cv_smm_bopf_results=None,
                n_jobs=8, drop_zero_variance=True, timestamp=None):
    classes = np.unique(labels)
    q_code = _pipeline.quantities_code()
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")

    base_file_name = "multi_ress_%s_%s.csv" % (q_code, timestamp)

    if cv_smm_bopf_results is None and wins is not None and wls is not None:
        cv_smm_bopf_results = cv_smm_bopf(data, labels, wins,
                                          wls, _pipeline,
                                          cv=cv, n_jobs=n_jobs,
                                          drop_zero_variance=drop_zero_variance)
    elif cv_smm_bopf_results is None and (wins is None or wls is None):
        raise ValueError("need to specify the CV_SMM_BOPF results or the values ofr wins and wls")

    data_mr_repr, cv_results, result_list, smm_best_acc, smm_best_data = cv_smm_bopf_results
    win_list = result_list["win"]
    wl_list = result_list["wl"]
    score_list = result_list["score"]

    if out_base_bopf_path is not None:
        try:
            save_base_bopf(data_mr_repr, win_list, wl_list, out_base_bopf_path)
            print("basic BOPF repr saved to %s" % out_base_bopf_path)
        except Exception as e:
            print("failed to save basic BOPF repr, error: ", e)

    # if out_path is not None:
    #     # save results of single_resolution
    #     save_first_stack(cv_results, win_list, wl_list, _pipeline, base_file_name, out_path)

    # rank results by acc
    rank_by_idx = np.argsort(score_list)[::-1]
    optimal_pairs = []
    optimal_acc = -1

    out_file = os.path.join(out_path, base_file_name)
    f = open(out_file, "a+")
    header = "top-k,stack,base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
    f.write(header)
    f.close()

    for ii, top_idx in enumerate(rank_by_idx[:top_k]):
        # incremental config
        x = data_mr_repr[win_list[top_idx]][wl_list[top_idx]]
        record_idxs = [top_idx]
        record_acc = score_list[top_idx]
        print("---> starting stack of configurations for top ", ii + 1)
        msg = "[(win:wl)"
        for used_idx in record_idxs:
            msg += "-(%.3f:%d)" % (win_list[used_idx], wl_list[used_idx])
        msg += "]"
        print("---> best config stack 1: %s ::::: acc: %.3f" % (msg, float(record_acc)))
        # stack up until resolution_max configs
        for jj in range(2, resolution_max + 1):
            try:
                print("---> starting search on stack level ", jj)
                # define an output file

                # define iter parameters to find best config
                best_config = None
                best_acc = record_acc

                # start iter to find next stack config
                for next_idx in rank_by_idx:
                    # if next_idx was not added to incremental config
                    if next_idx not in record_idxs:
                        # try to stack this idx and check acc
                        try:
                            # generate a compy of the repr matrix
                            x_i = sparse.hstack([x, data_mr_repr[win_list[next_idx]][wl_list[next_idx]]], format="csr")

                            # generate a custom message
                            message = "[(win:wl)"
                            for used_idx in record_idxs:
                                message += "-(%.3f:%d)" % (win_list[used_idx], wl_list[used_idx])
                            message += "-(%.3f:%d)]" % (win_list[next_idx], wl_list[next_idx])

                            # get a cr score from this current stack of configs
                            score, pipeline, dropped, shapes, cv_time = cv_score(x_i, labels, classes, _pipeline,
                                                                                 message=message, cv=cv, n_jobs=n_jobs)
                            # line to write to file
                            line = "%d,%d,%s,%d,%f,%d,%d,%d," % (
                                ii+1, jj, message, wl_list[next_idx], win_list[next_idx], dropped, shapes[0], shapes[1])

                            # if the iteration  was invalid (should always be valid here)
                            if pipeline is None:
                                # raise an error
                                raise ValueError("the cv failed for some reason")
                            # else
                            else:
                                # generate the rest of the line to write
                                if "lsa" in pipeline:
                                    exp_var = np.sum(pipeline["lsa"].explained_variance_ratio_)
                                else:
                                    exp_var = None
                                n_comps = _pipeline.K
                                scheme_name = pipeline["vsm"].get_scheme_notation()
                                mean_cv = float(np.mean(score))
                                std_cv = float(np.std(score))
                                line += "%f,%f,%f,%d,%s\n" % (mean_cv, std_cv,
                                                              exp_var if exp_var is not None else -1,
                                                              n_comps,
                                                              pipeline["vsm"].get_scheme_notation())

                                # and if this stack is the best so far, update variables
                                if mean_cv > best_acc:
                                    best_config = record_idxs + [next_idx]
                                    best_acc = mean_cv
                        except Exception as e:
                            print("failed on stack step k=%d, for wl=%d, win=%f, error: %s" % (
                                jj, wl_list[next_idx], win_list[next_idx], e))
                        else:

                            # write line to file
                            f = open(out_file, "a+")
                            try:
                                f.write(line)
                            except:
                                f.close()
                            else:
                                f.close()

                # after the iteration, if no improvement was achieve, stop the process
                if best_config is None or best_acc - record_acc < 0.01:  # at least improvement of 1%
                    print("ACCURACY IS NOT IMPROVING, STOPPING CODE")
                    break
                else:
                    # update the incremental configuration and advance to next stack level
                    x_arr = []
                    for used_idx in best_config:
                        x_arr.append(data_mr_repr[win_list[used_idx]][wl_list[used_idx]])
                    x = sparse.hstack(x_arr, format="csr")
                    record_idxs = best_config
                    record_acc = best_acc
                    msg = "[(win:wl)"
                    for used_idx in record_idxs:
                        msg += "-(%.3f:%d)" % (win_list[used_idx], wl_list[used_idx])
                    msg += "]"
                    print("---> best config stack %d: %s ::::: acc: %.3f" % (_pipeline.n, msg, float(record_acc)))
            except Exception as e:
                print("failed stack step k=%d, error: %s" % (_pipeline.n, e))

        if record_acc > optimal_acc:
            optimal_acc = record_acc
            optimal_pairs = record_idxs
    R = [(win_list[i], wl_list[i]) for i in optimal_pairs]
    print("FINAL BEST CONFIG: ", R, ", acc: ", optimal_acc)
    return R, timestamp, optimal_acc


def save_base_bopf(data_mr_repr, win_list, wl_list, out_base_bopf_path):
    metadata_dict = defaultdict(list)
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    for win, wl in zip(win_list, wl_list):
        sparse_repr_matrix = data_mr_repr[win][wl]
        out_file = os.path.join(
            out_base_bopf_path,
            "data_" + str(win) + "#" + str(round(wl, 3)) + "_" + time_stamp + ".npz")
        sparse.save_npz(out_file, sparse_repr_matrix)
        metadata_dict["win"].append(win)
        metadata_dict["wl"].append(wl)
        metadata_dict["file"].append(out_file)

    out_file = os.path.join(out_base_bopf_path, "metadata_" + time_stamp + ".csv")
    df = pd.DataFrame(metadata_dict, index=False)
    df.to_csv(out_file)


def save_first_stack(x_score, win_list, wl_list, _pipeline, base_file_name, out_path):
    out_file = os.path.join(out_path, base_file_name + "1-top_0-" + time.strftime(
        "%Y%m%d-%H%M%S") + ".csv")
    f = open(out_file, "a+")
    header = "base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
    f.write(header)
    f.close()

    msg = "[(win:wl)]"
    for i in range(len(win_list)):
        win = win_list[i]
        wl = wl_list[i]
        scores, cv_pipeline, dropped, shapes = x_score[win][wl]
        # line to write to file
        line = "%s,%d,%f,%d,%d,%d," % (msg, wl, win, dropped, shapes[0], shapes[1])
        exp_var = -1
        n_comps = _pipeline.K
        scheme = _pipeline.get_scheme_notation()
        mean_cv = -1
        std_cv = -1

        if cv_pipeline is not None:
            mean_cv = float(np.mean(scores))
            std_cv = float(np.std(scores))
            if "lsa" in cv_pipeline:
                exp_var = np.sum(cv_pipeline["lsa"].explained_variance_ratio_)

        line += "%f,%f,%f,%d,%s\n" % (mean_cv, std_cv, exp_var, n_comps, scheme)

        f = open(out_file, "a+")
        try:
            f.write(line)
        except:
            f.close()
        else:
            f.close()


def load_base_bopf(path, timestamp, _pipeline, labels, cv=5, n_jobs=8):
    data_mr_repr = defaultdict(lambda: defaultdict(object))
    classes = np.unique(labels)
    q_code = _pipeline.quantities_code()
    cv_results = defaultdict(lambda: defaultdict(object))
    result_lists = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*%s*" % timestamp)):
        win, wl = os.path.split(f)[1].split("_")[1].split("#")
        win = float(win)
        wl = int(wl)
        message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
        try:
            data_repr_i_full = sparse.load_npz(f)
            drop_zero_variance = VarianceThreshold()
            data_repr_i = drop_zero_variance.fit_transform(data_repr_i_full)
            print(":::::::: zero variance reduces features: ",
                  data_repr_i_full.shape, " -> ", data_repr_i.shape)
            data_mr_repr[win][wl] = data_repr_i
            cv_results_i = cv_score(data_repr_i, labels, classes, _pipeline, message=message, cv=cv, n_jobs=n_jobs)
            cv_results[win][wl] = cv_results_i
            result_lists["win"].append(win)
            result_lists["wl"].append(wl)

            if cv_results_i[1] is not None:
                result_lists["score"].append(np.mean(cv_results_i[0]))
            else:
                result_lists["score"].append(-1)
        except Exception as e:
            print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))

    return data_mr_repr, cv_results, result_lists


def load_bopf_from_quantity_search(path_data, path_cv_results, Q):
    data_mr_repr = defaultdict(lambda: defaultdict(object))
    cv_results = defaultdict(lambda: defaultdict(object))
    result_lists = defaultdict(list)

    df_cv_results = pd.read_csv(path_cv_results)
    df = df_cv_results[df_cv_results["quantity"] == Q]
    # set new wins and awls
    wins2 = np.unique(df["win"])
    wls2 = np.unique(df["wl"])
    Q_splitted = Q[1:-1].split("-")
    best_acc = -1
    best_data = None
    for wl in wls2:
        for win in wins2:
            try:
                bopf_data = []
                for q in Q_splitted:
                    f = os.path.join(path_data, "%s_%d_%.3f.npz" % (q, wl, win))
                    bopf_data.append(sparse.load_npz(f))
                data_i = sparse.hstack(bopf_data, format="csr")
                data_mr_repr[win][wl] = data_i
                # line = df.loc[(df["win2"] == int(win)) & (df["wl"] == wl)]
                line = df.loc[(df["win"] == win) & (df["wl"] == wl)]
                cv_results_i = ([line.cv_mean.values[0], line.cv_std.values[0]],
                                None, line.dropped.values[0],
                                [-1, -1], line.cv_time.values[0])
                cv_results[win][wl] = cv_results_i
                result_lists["win"].append(win)
                result_lists["wl"].append(wl)
                result_lists["score"].append(line.cv_mean.values[0])
                acc_i = line.cv_mean.values[0]
                if acc_i > best_acc:
                    best_acc = acc_i
                    best_data = (cv_results_i, win, wl)
            except Exception as e:
                print("failed load for [%d, %.3f] with error: " % (wl, win), win, e)
                raise e

    return data_mr_repr, cv_results, result_lists, best_acc, best_data


# scores, sklearn_pipeline, dropped, [-1, -1], end - ini
