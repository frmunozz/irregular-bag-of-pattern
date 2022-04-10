# -*- coding: utf-8 -*-
import os


if __name__ == '__main__':
    base_run = "python -m sklearnex avocado_knn_direct.py"

    for s1 in ["plasticc_train", "plasticc_augment_v3"]:
        for s2 in [True, False]:
            base_run2 = base_run + " %s plasticc_test --num_chunks=100" % s1
            if s2:
                base_run2 += " --use_metadata=True"
            print(base_run2)
            os.system(base_run2)  # no-prototype, no-normalize, no-scaler
            print(base_run2 + " --prototype=true")
            os.system(base_run2 + " --prototype=true")  # prototype no-normalize, no-scaler
            print(base_run2 + " --scaler=true")
            os.system(base_run2 + " --scaler=true")  # no-prototype, no-normalize, scaler
            print(base_run2 + " --normalizer=true")
            os.system(base_run2 + " --normalizer=true")  # no-prototype, normalize, no-scaler
            print(base_run2 + " --scaler=true --normalizer=true")
            os.system(base_run2 + " --scaler=true --normalizer=true")  # no-protoype, normalize, scaler

    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --prototype=true
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --scaler=true
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --normalizer=true
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --scaler=true --normalizer=true
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --prototype=true

    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --prototype=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --scaler=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --normalizer=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --scaler=true --normalizer=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --use_metadata=True
    # python avocado_knn_direct.py plasticc_augment_v3 plasticc_test --num_chunks=100 --prototype=true --use_metadata=True

    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --prototype=true
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --scaler=true
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --normalizer=true
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --scaler=true --normalizer=true
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --prototype=true

    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --prototype=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --scaler=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --normalizer=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --scaler=true --normalizer=true --use_metadata=True
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --use_metadata=True
    # python avocado_knn_direct.py plasticc_train plasticc_test --num_chunks=100 --prototype=true --use_metadata=True