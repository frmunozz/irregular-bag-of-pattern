# -*- coding: utf-8 -*-

if __name__ == '__main__':
    # read dataset
    dataset, labels_, metadata, split_folds = gen_dataset_from_h5(args.dataset, bands=_BANDS, num_folds=5)
    split_folds = rearrange_splits(split_folds)
    classes = np.unique(labels_)
    print(len(labels_))
    # estimate spatial complexity
    sc = int(np.mean([len(ts[0]) * 2 for ts in dataset]))
    print("the estimated size of each time series is {} [4 bytes units] in average".format(sc))

    # estimate max window of observation
    time_durations = np.array(
        [ts[1][-1] - ts[1][0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)

    # define some fixed parameters
    alpha = args.alpha

    """ (MmTrMn-VaMmMx-TrMnMe) """
    # Q = [["min_max", "trend", "min"], ["var", "min_max", "max"], ["trend", "min", "mean"]]
    # q_search_path = os.path.join("..", "data", "quantity_search", "triple_quantity_lsa_data")
    # q_search_cv_results_file = os.path.join("..", "data", "quantity_search",
    #                                         "comb_triple_quantity_lsa_20210904-024910")
    """"""

    """" (MmTr-MmMn-MmMx-TrMn-VaMn) """
    # Q = [["min_max", "trend"], ["min_max", "min"], ["min_max", "max"], ["trend", "min"], ["var", "min"]]
    # q_search_path = os.path.join("..", "data", "quantity_search", "double_quantity_lsa_data")
    # q_search_cv_results_file = os.path.join("..", "data", "quantity_search",
    #                                         "comb_double_quantity_lsa_20210904-024910")
    """"""

    doc_kwargs = {
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }

    lsa_kwargs = {  # scheme: ltc
        "class_based": False,  # options: True, False
        "normalize": "l2",  # options: None, l2
        "use_idf": True,  # options: True, False
        "sublinear_tf": True  # options: True, False
    }

    wls = [1, 2, 3]
    wins = np.logspace(np.log10(30), np.log10(mean_time + std_time * 2), 20)
    C = args.compact_method  # compact method
    N = sc
    resolution_max = 4  # fixed
    top_k = args.top_k
    n_jobs = 6  # fixed
    pre_load_bopf = True
    drop_zero_variance = True

    # output paths'
    if not os.path.exists(os.path.join("..", "data", "configs_results_new")):
        os.mkdir(os.path.join("..", "data", "configs_results_new"))

    out_path = os.path.join("..", "data", "configs_results_new", "%s_multi_ress_search" % C.lower())
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_base_bopf_path = None
    # out_base_bopf_path = os.path.join("..", "data", "configs_results_new", "full_raw_bopf")
    # if not os.path.exists(out_base_bopf_path):
    #     os.mkdir(out_base_bopf_path)

    if not os.path.exists(os.path.join("..", "data", "configs_results_new", "optimal_config")):
        os.mkdir(os.path.join("..", "data", "configs_results_new", "optimal_config"))

    config_file_path = os.path.join("..", "data", "configs_results_new", "optimal_config", C.lower())
    if not os.path.exists(config_file_path):
        os.mkdir(config_file_path)

    if not os.path.exists(out_path):
        raise ValueError("folder {} doesnt exists".format(out_path))
    # if not os.path.exists(out_base_bopf_path):
    #     raise ValueError("folder {} doesnt exists".format(out_base_bopf_path))
    if not os.path.exists(config_file_path):
        raise ValueError("file {} doesnt exists".format(config_file_path))
    config_file = os.path.join(config_file_path, "config_numba.json")

    # get pipeline
    method = MMMBOPF(alpha=alpha, Q=Q, C=C, lsa_kw=lsa_kwargs,
                     doc_kw=doc_kwargs, N=N, n_jobs=n_jobs,
                     drop_zero_variance=drop_zero_variance)

    # testing bug
    # data_repr_i = pipeline.multi_quantity_representation(dataset, 90.567, 3)
    # drop_zero_variance = VarianceThreshold()
    # data_repr_i2 = drop_zero_variance.fit_transform(data_repr_i)
    # print(data_repr_i.shape, data_repr_i2.shape)
    # cv_results_i = cv_score(data_repr_i2, labels_, classes, pipeline,
    #                         message="testing", cv=split_folds, n_jobs=n_jobs)
    # raise ValueError("force stop")

    # pre-load saved base bopf
    if pre_load_bopf:
        print("LOADING PRECOMPUTED BASE BOPF...")
        # timestamp = "20210812-165647"
        # cv_smm_bopf_results = load_base_bopf(out_base_bopf_path, timestamp, pipeline,
        #                                     labels_, cv=split_folds, n_jobs=n_jobs)
        cv_smm_bopf_results = load_bopf_from_quantity_search(args.q_search_path,
                                                             args.q_search_cv_results_file,
                                                             method.quantities_code(), wls, wins)
    else:
        cv_smm_bopf_results = None
    ini = time.time()
    R, timestamp, optimal_acc = cv_mmm_bopf(dataset, labels_, method, cv=split_folds,
                                            resolution_max=resolution_max, top_k=top_k, out_path=out_path,
                                            out_base_bopf_path=out_base_bopf_path, n_jobs=n_jobs,
                                            cv_smm_bopf_results=cv_smm_bopf_results,
                                            drop_zero_variance=drop_zero_variance)
    end = time.time()