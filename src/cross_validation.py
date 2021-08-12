import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import time
from scipy import sparse


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
        return [-1], None, dropped, [-1, -1]

    n_variables = len(_BANDS)
    n_observations, m_temp = x.shape
    n_features = m_temp // n_variables
    print("initial matrix shape:", x.shape,
          "(n_observations: %d, n_features: %d, n_variables: %d)" % (n_observations,
                                                                     n_features,
                                                                     n_variables))
    print("target number of features:",  _pipeline.n * len(_BANDS))
    sklearn_pipeline = _pipeline.get_sklearn_pipeline(n_variables, n_features, classes)

    print("[%s]: DOING CROSS VALIDATION..." % message, end="\r")
    ini = time.time()
    scores = cross_val_score(sklearn_pipeline, x, labels,
                             scoring="balanced_accuracy", cv=cv, n_jobs=n_jobs, verbose=0)
    end = time.time()
    print("[%s]: %.3f += %.3f (time: %.3f sec)" % (message, float(np.mean(scores)),
                                                   float(np.std(scores)), end - ini))

    return scores, sklearn_pipeline, dropped, [-1, -1]


def cv_smm_bopf(data, labels, wins, wls, _pipeline, cv=5, n_jobs=8):
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
    for wl in wls:
        for win in wins:
            message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
            try:
                data_repr_i = _pipeline.multi_quantity_representation(data, win, wl)
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


def cv_mmm_bopf(data, labels, wins, wls, _pipeline, cv=5, resolution_max=4, top_k=4,
                out_path=None, cv_smm_bopf_results=None, n_jobs=8):
    classes = np.unique(labels)
    q_code = _pipeline.quantities_code()
    base_file_name = "multi_ress-%s-stack_" % q_code

    if cv_smm_bopf_results is None:
        cv_smm_bopf_results = cv_smm_bopf(data, labels, wins, wls, _pipeline, cv=cv, n_jobs=n_jobs)

    data_mr_repr, cv_results, result_list = cv_smm_bopf_results
    win_list = result_list["win"]
    wl_list = result_list["wl"]
    score_list = result_list["score"]

    if out_path is not None:
        # save results of single_resolution
        save_first_stack(cv_results, win_list, wl_list, _pipeline, base_file_name, out_path)

    # rank results by acc
    rank_by_idx = np.argsort(score_list)[::-1]

    for ii, top_idx in enumerate(rank_by_idx[:top_k]):
        # incremental config
        x = data_mr_repr[win_list[top_idx]][wl_list[top_idx]]
        record_idxs = [top_idx]
        record_acc = score_list[top_idx]
        print("---> starting stack of configurations for top ", ii + 1)
        msg = "[{win,wl}"
        for used_idx in record_idxs:
            msg += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
        msg += "]"
        print("---> best config stack 1: %s ::::: acc: %.3f" % (msg, float(record_acc)))
        # stack up until resolution_max configs
        for jj in range(2, resolution_max + 1):
            try:
                print("---> starting search on stack level ", jj)
                # define an output file
                out_file = os.path.join(out_path,
                                        base_file_name + str(jj) + "-top_" + str(ii + 1) + "-" + time.strftime(
                                            "%Y%m%d-%H%M%S") + ".csv")
                f = open(out_file, "a+")
                header = "base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
                f.write(header)
                f.close()

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
                            message = "{win,wl}"
                            for used_idx in record_idxs:
                                message += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
                            message += ",{%.3f,%d}" % (win_list[next_idx], wl_list[next_idx])

                            # get a cr score from this current stack of configs
                            score, pipeline, dropped, shapes = cv_score(x_i, labels, classes, _pipeline,
                                                                                   message=message, cv=cv, n_jobs=n_jobs)
                            # line to write to file
                            line = "%s,%d,%f,%d,%d,%d," % (
                            message, wl_list[next_idx], win_list[next_idx], dropped, shapes[0], shapes[1])

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
                if best_config is None or best_acc <= record_acc:
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
                    msg = "[{win,wl}"
                    for used_idx in record_idxs:
                        msg += ",{%.3f,%d}" % (win_list[used_idx], wl_list[used_idx])
                    msg += "]"
                    print("---> best config stack %d: %s ::::: acc: %.3f" % (_pipeline.n, msg, float(record_acc)))
            except Exception as e:
                print("failed stack step k=%d, error: %s" % (_pipeline.n, e))


def save_first_stack(x_score, win_list, wl_list, _pipeline, base_file_name, out_path):
    out_file = os.path.join(out_path, base_file_name + "1-top_0-" + time.strftime(
        "%Y%m%d-%H%M%S") + ".csv")
    f = open(out_file, "a+")
    header = "base,wl,win,dropped,shape_before,shape_after,mean_cv,std_cv,exp_var,n_comp,scheme\n"
    f.write(header)
    f.close()

    msg = "[{win,wl}]"
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