# -*- coding: utf-8 -*-
import os

if __name__ == '__main__':
    base_run = "python -m sklearnex mmmbopf_knn_direct.py"

    # c = "LSA"
    c = "MANOVA"

    for s1 in ["plasticc_augment_v3", "plasticc_train"]:
        for s2 in [True, False]:
            base_run2 = base_run + " %s plasticc_test --num_chunks=100" % s1
            if s2:
                base_run2 += " --use_metadata=True"
            print(base_run2 + " --tag=features_%s" % c)
            os.system(base_run2 + " --tag=features_%s" % c)  # no-prototype, no-normalize, no-scaler
            print(base_run2 + " --prototype=true --tag=features_%s" % c)
            os.system(base_run2 + " --prototype=true --tag=features_%s" % c)  # prototype no-normalize, no-scaler
            print(base_run2 + " --scaler=true --tag=features_%s" % c)
            os.system(base_run2 + " --scaler=true --tag=features_%s" % c)  # no-prototype, no-normalize, scaler
            print(base_run2 + " --normalizer=true --tag=features_%s" % c)
            os.system(base_run2 + " --normalizer=true --tag=features_%s" % c)  # no-prototype, normalize, no-scaler
            print(base_run2 + " --scaler=true --normalizer=true --tag=features_%s" % c)
            os.system(base_run2 + " --scaler=true --normalizer=true --tag=features_%s" % c)  # no-protoype, normalize, scaler