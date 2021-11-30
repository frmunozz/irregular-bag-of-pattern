# -*- coding: utf-8 -*-
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_path)
import numpy as np
import argparse
import pandas as pd


def add_count_and_file_Qs(df, results_file):
    df.loc[:, "count_Qs"] = [len(x.split("-")) for x in df["quantity"]]
    df.loc[:, "cv_results_file"] = [results_file] * len(df)
    return df


def comb_filter_only_bests(df):
    df2 = df[df["count_Qs"] == 2]
    cv_mean = df2["cv_mean"].to_numpy()
    highest_idx = np.argmax(cv_mean)
    line = df2.iloc[highest_idx]
    df_filter = ((df["count_Qs"] == 2) & (df2["quantity"] == line.quantity))
    n_max = np.max(df["count_Qs"])
    for i in range(3, n_max):  # we avoid last iteration since it doesnt improve in more than 1%
        df2 = df[df["count_Qs"] == i]
        cv_mean = df2["cv_mean"].to_numpy()
        highest_idx = np.argmax(cv_mean)
        line = df2.iloc[highest_idx]
        df_filter |= ((df["count_Qs"] == i) & (df2["quantity"] == line.quantity))

    return df[df_filter]


def concat_single_best(df1, df2):
    cv_mean = df1["cv_mean"].to_numpy()
    highest_idx = np.argmax(cv_mean)
    highest_Q = df1.iloc[highest_idx].quantity
    df3 = df1[df1["quantity"] == highest_Q]
    return pd.concat([df3, df2])


def resume_quantity_search(dfs):
    df1 = dfs[0].reset_index()
    df2 = dfs[1].reset_index()
    df3 = dfs[2].reset_index()

    qs1 = np.unique(df1["quantity"])
    qs2 = np.unique(df2["quantity"])
    qs3 = np.unique(df3["quantity"])

    idxs = []
    for q in qs1:
        df_tmp = df1[df1["quantity"] == q]
        high_idx = np.argmax(df_tmp["cv_mean"])
        idxs.append(high_idx)
    df1_bests = df1.iloc[idxs]
    df1_bests.loc[:, "type"] = np.array(["Single-Q"] * len(df1_bests))

    idxs = []
    for q in qs2:
        high_idx = np.argmax(df2[df2["quantity"] == q]["cv_mean"])
        idxs.append(high_idx)
    df2_bests = df2.iloc[idxs]
    df2_bests.loc[:, "type"] = np.array(["Double-Q"] * len(df2_bests))

    idxs = []
    for q in qs3:
        high_idx = np.argmax(df3[df3["quantity"] == q]["cv_mean"])
        idxs.append(high_idx)
    df3_bests = df3.iloc[idxs]
    df3_bests.loc[:, "type"] = np.array(["Triple-Q"] * len(df3_bests))

    res_df = pd.concat([df1_bests, df2_bests, df3_bests])
    return res_df[["quantity", "type", "cv_mean", "cv_std", "win", "wl",
                   "alpha", "dropped", "bopf_shape", "cv_time",
                   "q_search_path", "cv_results_file"]]


def create_resume_file(C, timestamp, out_path):

    dfs = []
    for key in ["single", "double", "triple"]:
        comb_quantity_file = os.path.join(out_path, "comb_%s_quantity_%s_%s" % (key, C.lower(), timestamp))
        quantity_file = os.path.join(out_path, "%s_quantity_%s_%s" % (key, C.lower(), timestamp))
        data_folder = os.path.join(out_path, "%s_quantity_%s_data" % (key, C.lower()))

        df_q = pd.read_csv(quantity_file, index_col=None)
        df_q = df_q[df_q["valid_cv"]]
        add_count_and_file_Qs(df_q, quantity_file)

        df_comb_q = pd.read_csv(comb_quantity_file, index_col=None)
        df_comb_q = df_comb_q[df_comb_q["valid_cv"]]
        add_count_and_file_Qs(df_comb_q, comb_quantity_file)

        df_comb_q2 = comb_filter_only_bests(df_comb_q)
        df_comb_q2 = concat_single_best(df_q, df_comb_q2)
        df_comb_q2.loc[:, "q_search_path"] = [data_folder] * len(df_comb_q2)
        dfs.append(df_comb_q2)

    resume_df = resume_quantity_search(dfs)
    resume_file = os.path.join(out_path, "quantity_search_resume.csv")
    resume_df.to_csv(resume_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "compact_method",
        help="The compact method to use, options are: LSA or MANOVA"
    )
    parser.add_argument(
        "timestamp",
        help="timestamp for creating unique files"
    )
    parser.add_argument(
        "out_path",
        help="Directory to where will be saved the resume file"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of process to run in parallel"
    )

    args = parser.parse_args()

    create_resume_file(args.compact_method, args.timestamp, args.out_path)
