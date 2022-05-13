import os
import sys
import argparse
import subprocess

base_train = ["python", "scripts/ibopf_lgbm_train_classifier.py", "plasticc_augment_v3", "--use_metadata"]
base_predict = ["python", "scripts/ibopf_lgbm_predict.py", "plasticc_test"]


def calls(classifier, tag, only_var_star, only_supernova, num_chunks):
    local_add = [
        "--classifier", classifier,
        "--tag", tag]
    if only_var_star:
        local_add.append("--only_var_star")
    elif only_supernova:
        local_add.append("--only_supernova")

    train_call = base_train.copy()
    train_call.extend(local_add)

    predict_call = base_predict.copy()
    predict_call.extend(local_add)
    predict_call.extend(["--num_chunks", str(num_chunks)])

    return train_call, predict_call


def launch_subprocess(cc):
    for c in cc:
        print(" ".join(c))
        try:
            subprocess.check_call(c)
        except subprocess.CalledProcessError:
            print("################################################")
            print("error while runing process, ending")
            print("################################################")
            return False
    return True


def main():
    tag = "features_v4_LSA_361"
    classifier = "lgbmExtraSubset"
    for case in [False, True]:
        c1, c2 = calls(classifier, tag, case, not case, 25)
        cc = [c1, c2]
        ok = launch_subprocess(cc)
        if not ok:
            # we stop the experiment
            return


def main_full():
    tag = "features_v4_LSA"
    classifier = "lgbmExtraFull"
    for case in [False, True]:
        c1, c2 = calls(classifier, tag, case, not case, 100)
        cc = [c1, c2]
        ok = launch_subprocess(cc)
        if not ok:
            # we stop the experiment
            return


if __name__ == '__main__':
    main()
    main_full()
