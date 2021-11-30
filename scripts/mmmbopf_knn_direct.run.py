# -*- coding: utf-8 -*-
import os



if __name__ == '__main__':
    base_run = "python -m sklearnex mmmbopf_knn_direct.py"

    for s1 in ["plasticc_augment_v3", "plasticc_train"]:
        for s2 in [False]:
            base_run2 = base_run + " %s plasticc_test --num_chunks=100" % s1
            if s2:
                base_run2 += " --use_metadata=True"
            # print(base_run2 + " --tag=features_LSA")
            # os.system(base_run2 + " --tag=features_LSA")  # no-prototype, no-normalize, no-scaler
            # print(base_run2 + " --prototype=true --tag=features_LSA")
            # os.system(base_run2 + " --prototype=true --tag=features_LSA")  # prototype no-normalize, no-scaler
            print(base_run2 + " --scaler=true --tag=features_LSA")
            os.system(base_run2 + " --scaler=true --tag=features_LSA")  # no-prototype, no-normalize, scaler
            # print(base_run2 + " --normalizer=true --tag=features_LSA")
            # os.system(base_run2 + " --normalizer=true --tag=features_LSA")  # no-prototype, normalize, no-scaler
            # print(base_run2 + " --scaler=true --normalizer=true --tag=features_LSA")
            # os.system(base_run2 + " --scaler=true --normalizer=true --tag=features_LSA")  # no-protoype, normalize, scaler