import argparse
import numpy as np

import avocado
import os
import sys
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, main_path)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import time
from scipy import sparse

from src.preprocesing import gen_dataset_from_h5
from src.feature_extraction.text import MPTextGenerator
from src.feature_extraction.vector_space_model import VSM
from src.feature_extraction.centroid import CentroidClass
from sklearn.decomposition import TruncatedSVD
from src.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

_BANDS = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
_N_JOBS = 6

symbols = {
        "mean": "M",
        "std": "S",
        "trend": "T",
        "min_max": "m",
}

merged_labels_to_num = {
    "Single microlens": 1,
    "TDE": 2,
    "SN": 3,
    "SLSN": 4,
    "M-dwarf": 5,
    "AGN": 6,
    "EB": 7,
    "Mira": 8,
    "RRL": 9,
    "Unknown": 99
}

merged_labels_to_str = {
    1: "Single microlens",
    2: "TDE",
    3: "SN",
    4: "SLSN",
    5: "M-dwarf",
    6: "AGN",
    7: "EB",
    8: "Mira",
    9: "RRL",
    99: "Unknown",
}


merged_labels2 = {
    6: "Single microlens",
    15: "TDE",
    16: "Short period VS",
    42: "SN",
    52: "SN",
    53: "Short period VS",
    62: "SN",
    64: "SN",
    65: "M-dwarf",
    67: "SN",
    88: "AGN",
    90: "SN",
    92: "Short period VS",
    95: "SN",
    99: "Unknown"
}

merged_labels = {
    6: "Single microlens",
    15: "TDE",
    16: "EB",
    42: "SN",
    52: "SN",
    53: "Mira",
    62: "SN",
    64: "SN",
    65: "M-dwarf",
    67: "SN",
    88: "AGN",
    90: "SN",
    92: "RRL",
    95: "SN",
    99: "Unknown"
}

labels_to_str= {
    6: "Single microlens",
    15: "TDE",
    16: "EB",
    42: "SNII",
    52: "SNIax",
    53: "Mira",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RRL",
    95: "SLSN-I",
    99: "Unknown"
}

ordered_classes = [42, 52, 62, 64, 67, 90, 95, 16, 53, 92, 6, 15, 65, 88]


def cv_score(features, labels, classes, class_based, text='X'):
    class_based = False
    ini = time.time()
    centroid = CentroidClass(classes=classes)
    knn = KNeighborsClassifier(n_neighbors=1, classes=classes, useClasses=class_based)
    pipeline = Pipeline([("knn", knn)])
    scores = cross_val_score(pipeline, features, labels, scoring="balanced_accuracy", cv=5, n_jobs=6, verbose=1)
    # scores = None
    end = time.time()
    print("[%s]:" % text, np.mean(scores), "+-", np.std(scores), " (time: %.3f sec)" % (end - ini))

    y_pred = cross_val_predict(pipeline, features, labels, cv=5, n_jobs=6, verbose=1)
    return scores, pipeline, y_pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="YlGnBu"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def quantities_code(quantities):
    f_name = ""
    for q_i in quantities:
        for q_ii in q_i:
            f_name += symbols[q_ii]
        f_name += "-"
    return f_name


def get_all_single_representations(data, labels, wins, wls, alphabet_size, quantities, classes, doc_kwargs):
    q_code = quantities_code(quantities)
    x_repr = defaultdict(lambda: defaultdict(object))
    for wl, win in zip(wls, wins):
        if True:
            message = "[win: %.3f, wl: %d, q: %s]" % (win, wl, q_code)
            try:
                x_i = multi_quantity_representation(data, win, wl, alphabet_size, quantities, doc_kwargs)
                x_repr[win][wl] = x_i
            except Exception as e:
                print("failed iteration wl=%d, win=%f, error: %s" % (wl, win, e))

    return x_repr


def multi_quantity_representation(data,  win, wl, alphabet_size, q, doc_kwargs):
    if len(q) > 1:
        x_arr = []
        for q_i in q:
            doc_kwargs["quantity"] = np.array(q_i)
            doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q_i))
            text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                       opt_desc=", " + "-".join(q_i), **doc_kwargs)
            x_i = text_gen.fit_transform(data)
            x_arr.append(x_i)
        x = sparse.hstack(x_arr, format="csr")
    else:
        doc_kwargs["quantity"] = np.array(q[0])
        doc_kwargs["alphabet_size"] = np.array([alphabet_size] * len(q[0]))
        text_gen = MPTextGenerator(bands=_BANDS, n_jobs=_N_JOBS, win=win, wl=wl, direct_bow=True, tol=wl * 2,
                                   opt_desc=", " + "-".join(q[0]), **doc_kwargs)
        x = text_gen.fit_transform(data)

    return x


def our_method_features(set_name, n_components=300):
    file = "our_method_features_%d.npy" % n_components
    if os.path.exists(os.path.join("..", file)):
        print("features saved, loading")
        return np.load(file, allow_pickle=True)

    dataset, labels_, metadata = gen_dataset_from_h5(set_name)
    classes = np.unique(labels_)
    time_durations = np.array(
        [ts.observations["time"].to_numpy()[-1] - ts.observations["time"].to_numpy()[0] for ts in dataset])
    mean_time = np.mean(time_durations)
    std_time = np.std(time_durations)
    max_window = mean_time + std_time

    doc_kwargs = {
        "irr_handler": "#",
        "mean_bp_dist": "normal",
        "verbose": True,
    }

    lsa_kwargs = {
        "class_based": True,  # options: True, False
        "normalize": "l2",  # options: None, l2
        "use_idf": True,  # options: True, False
        "sublinear_tf": True  # options: True, False
    }

    alphabet_size = 6
    q = [["mean", "trend", "min_max"]]

    wls = [1, 1, 2, 1, 2]
    wins = [94.472, 1047.782, 400.206, 110.909, 892.496]

    x_repr = get_all_single_representations(dataset, labels_, wins, wls, alphabet_size,
                                            q, classes, doc_kwargs)
    x_arr = []
    for wl, win in zip(wls, wins):
        x_arr.append(x_repr[win][wl])
    x_i = sparse.hstack(x_arr, format="csr")
    sel = VarianceThreshold(threshold=0)
    x_i = sel.fit_transform(x_i)
    vsm = VSM(class_based=False, classes=classes, norm=lsa_kwargs["normalize"], use_idf=lsa_kwargs["use_idf"],
              smooth_idf=True, sublinear_tf=lsa_kwargs["sublinear_tf"])
    x_i = vsm.fit_transform(x_i, y=labels_)

    lsa = TruncatedSVD(n_components=n_components)
    x_i = lsa.fit_transform(x_i, y=labels_)
    print("saving the features to disc")
    np.save("our_method_features_%d.npy" % n_components, x_i)

    return x_i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset '%s'..." % args.dataset)
    dataset = avocado.load(args.dataset, metadata_only=True)
    labels = dataset.metadata['class'].to_numpy()
    classes = np.unique(labels)

    dataset2, labels2_, metadata2 = gen_dataset_from_h5(args.dataset)
    print("verify labels:", all(np.array(labels) == np.array(labels2_)))

    # Load the dataset raw features
    print("Loading raw features...")
    dataset.load_raw_features()
    df = dataset.raw_features

    print("Preprocessing raw features...")

    # process features
    features_i = dataset.select_features(avocado.plasticc.PlasticcFeaturizer())

    # drop metadata from features
    host_photoz = features_i["host_photoz"].to_numpy()
    host_photoz_error = features_i["host_photoz_error"].to_numpy()
    features_metadata = np.hstack((host_photoz.reshape(-1,1), host_photoz_error.reshape(-1,1)))
    print("metadata features shape:", features_metadata.shape)
    features_i = features_i.drop(columns=["host_photoz", "host_photoz_error"])


    # drop nan columns
    nan_columns = features_i.columns[features_i.isnull().any()].tolist()
    print("nan_columns:", nan_columns)
    features_i = features_i.drop(columns=nan_columns)

    # transform to matrix
    features_i = features_i.to_numpy(dtype=float)

    # scale
    scale = StandardScaler()
    features_avocado = scale.fit_transform(features_i)
    feature_metadata = scale.fit_transform(features_metadata)

    # add features from our method
    print("computing features from our method")
    features_our = our_method_features(args.dataset, n_components=500)

    # stack features
    features_combined = np.hstack((features_avocado, features_our, feature_metadata))

    # pca = PCA(n_components=300)
    # print("doing pca to combined features")
    # features_combined = pca.fit_transform(features_combined, y=labels)

    features_our = np.hstack((features_our, feature_metadata))
    # pca2 = PCA()
    # features_our = pca2.fit_transform(features_our)
    # print("shape feature our after add metadata:", features_our.shape)

    features_avocado = np.hstack((features_avocado, feature_metadata))

    print("our features shape:", features_our.shape)
    print("avocado features shape:", features_avocado.shape)
    print("combined features shape:", features_combined.shape)


    # perform 1-NN classified on cross validation and using centroid classes
    ############################################################

    m_labels = np.array([merged_labels_to_num[merged_labels[x]] for x in labels])
    m_classes = np.unique(m_labels)
    m_classes_names = np.array([merged_labels_to_str[x] for x in m_classes])

    ordered_classes_names = np.array([labels_to_str[x] for x in ordered_classes])
    m_ordered_classes = np.unique(np.array([merged_labels_to_num[merged_labels[x]] for x in ordered_classes]))
    m_ordered_classes_names = np.array([merged_labels_to_str[x] for x in m_ordered_classes])

    # out_path = os.path.join(main_path, "data", "conf_matrix", "plasticc_train", "knn")
    # for fea, fea_name in zip([features_our, features_avocado, features_combined],
    #                          ["our", "avocado", "combined"]):
    #     print("Starting classifier %s features KNN..." % fea_name)
    #     _, _, y_pred = cv_score(fea, labels, classes, True, text="our features 1NN")
    #     print("full classes acc:", balanced_accuracy_score(labels, y_pred))
    #     m_pred = np.array([merged_labels_to_num[merged_labels[x]] for x in y_pred])
    #     print("merged SN classes acc:", balanced_accuracy_score(m_labels, m_pred))
    #
    #     # plot full classes
    #     conf = confusion_matrix(labels, y_pred, labels=ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=False,
    #                       title='Conf. matrix %s features KNN [b_acc:%.3f]' % (fea_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_knn.png" % fea_name), dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=True,
    #                       title='Conf. matrix %s features KNN [b_acc:%.3f]' % (fea_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_knn_normed.png" % fea_name), dpi=300)
    #
    #     # plot SN merged classes
    #     conf = confusion_matrix(m_labels, m_pred, labels=m_ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=False,
    #                           title='Conf. matrix %s features KNN [b_acc:%.3f]' % (fea_name, balanced_accuracy_score(m_labels,
    #                                                                                                        m_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_knn_merged.png" % fea_name), dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=True,
    #                           title='Conf. matrix %s features KNN [b_acc:%.3f]' % (fea_name, balanced_accuracy_score(m_labels,
    #                                                                                                        m_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_knn_normed_merged.png" % fea_name), dpi=300)
    #
    #     print("Done!")
    ##############################################################################
    ##############################################################################

    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        # "learning_rate": 0.05,
    #     "colsample_bytree": 0.5,
    #     "reg_alpha": 0.0,
    #     "reg_lambda": 0.0,
    #     "min_split_gain": 10.0,
    #     "min_child_weight": 2000.0,
    #     "n_estimators": 5000,
    #     "silent": -1,
    #     "verbose": -1,
    #     "max_depth": 7,
    #     "num_leaves": 50,
    }
    # classifier_name = "LGBM"
    # out_path = os.path.join(main_path, "data", "conf_matrix", "plasticc_train", "lgbm")
    # for fea, fea_name in zip([features_our, features_avocado, features_combined],
    #                          ["our", "avocado", "combined"]):
    #     classifier = lgb.LGBMClassifier(**lgb_params)
    #     print("Starting classifier %s features %s..." % (fea_name, classifier_name))
    #     y_pred = cross_val_predict(classifier, fea, labels, cv=5, n_jobs=6, verbose=1)
    #     print("full classes acc:", balanced_accuracy_score(labels, y_pred))
    #     m_pred = np.array([merged_labels_to_num[merged_labels[x]] for x in y_pred])
    #     print("merged SN classes acc:", balanced_accuracy_score(m_labels, m_pred))
    #
    #     # plot full classes
    #     conf = confusion_matrix(labels, y_pred, labels=ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=False,
    #                       title='Conf. matrix %s features %s [b_acc:%.3f]' % (fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s.png" % (fea_name, classifier_name)), dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=True,
    #                       title='Conf. matrix %s features %s [b_acc:%.3f]' % (fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s_normed.png" % (fea_name, classifier_name)), dpi=300)
    #
    #     # plot SN merged classes
    #     conf = confusion_matrix(m_labels, m_pred, labels=m_ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=False,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (fea_name, classifier_name, balanced_accuracy_score(m_labels,
    #                                                                                                        m_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s_merged.png" % (fea_name, classifier_name)), dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=True,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (fea_name, classifier_name, balanced_accuracy_score(m_labels,
    #                                                                                                        m_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s_normed_merged.png" % (fea_name, classifier_name)), dpi=300)
    #
    #     print("Done!")

    ##############################################################################
    ##############################################################################

    classifier_name = "SVM"
    out_path = os.path.join(main_path, "data", "conf_matrix", "plasticc_train", "svm")
    for fea, fea_name in zip([features_our, features_avocado, features_combined],
                             ["our", "avocado", "combined"]):
        classifier = SVC(kernel="linear")
        print("Starting classifier %s features %s..." % (fea_name, classifier_name))
        y_pred = cross_val_predict(classifier, fea, labels, cv=5, n_jobs=6, verbose=1)
        print("full classes acc:", balanced_accuracy_score(labels, y_pred))
        m_pred = np.array([merged_labels_to_num[merged_labels[x]] for x in y_pred])
        print("merged SN classes acc:", balanced_accuracy_score(m_labels, m_pred))

        # plot full classes
        conf = confusion_matrix(labels, y_pred, labels=ordered_classes)
        fig = plt.figure(figsize=(12, 10))
        plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=False,
                              title='Conf. matrix %s features %s [b_acc:%.3f]' % (
                              fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
        plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s.png" % (fea_name, classifier_name)),
                    dpi=300)

        fig = plt.figure(figsize=(12, 10))
        plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=True,
                              title='Conf. matrix %s features %s [b_acc:%.3f]' % (
                              fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
        plt.savefig(
            os.path.join(out_path, "conf_matrix_%s_features_%s_normed.png" % (fea_name, classifier_name)),
            dpi=300)

        # plot SN merged classes
        conf = confusion_matrix(m_labels, m_pred, labels=m_ordered_classes)
        fig = plt.figure(figsize=(12, 10))
        plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=False,
                              title='Conf. matrix %s features %s [b_acc:%.3f]' % (
                              fea_name, classifier_name, balanced_accuracy_score(m_labels,
                                                                                 m_pred)))
        plt.savefig(
            os.path.join(out_path, "conf_matrix_%s_features_%s_merged.png" % (fea_name, classifier_name)),
            dpi=300)

        fig = plt.figure(figsize=(12, 10))
        plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=True,
                              title='Conf. matrix %s features %s [b_acc:%.3f]' % (
                              fea_name, classifier_name, balanced_accuracy_score(m_labels,
                                                                                 m_pred)))
        plt.savefig(os.path.join(out_path,
                                 "conf_matrix_%s_features_%s_normed_merged.png" % (fea_name, classifier_name)), dpi=300)

        print("Done!")

    ##############################################################################
    ##############################################################################

    # classifier_name = "RF"
    # out_path = os.path.join(main_path, "data", "conf_matrix", "plasticc_train", "rf")
    # for fea, fea_name in zip([features_our, features_avocado, features_combined],
    #                          ["our", "avocado", "combined"]):
    #     classifier = RandomForestClassifier()
    #     print("Starting classifier %s features %s..." % (fea_name, classifier_name))
    #     y_pred = cross_val_predict(classifier, fea, labels, cv=5, n_jobs=6, verbose=1)
    #     print("full classes acc:", balanced_accuracy_score(labels, y_pred))
    #     m_pred = np.array([merged_labels_to_num[merged_labels[x]] for x in y_pred])
    #     print("merged SN classes acc:", balanced_accuracy_score(m_labels, m_pred))
    #
    #     # plot full classes
    #     conf = confusion_matrix(labels, y_pred, labels=ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=False,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (
    #                               fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(os.path.join(out_path, "conf_matrix_%s_features_%s.png" % (fea_name, classifier_name)),
    #                 dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=ordered_classes_names, normalize=True,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (
    #                               fea_name, classifier_name, balanced_accuracy_score(labels, y_pred)))
    #     plt.savefig(
    #         os.path.join(out_path, "conf_matrix_%s_features_%s_normed.png" % (fea_name, classifier_name)),
    #         dpi=300)
    #
    #     # plot SN merged classes
    #     conf = confusion_matrix(m_labels, m_pred, labels=m_ordered_classes)
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=False,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (
    #                               fea_name, classifier_name, balanced_accuracy_score(m_labels,
    #                                                                                  m_pred)))
    #     plt.savefig(
    #         os.path.join(out_path, "conf_matrix_%s_features_%s_merged.png" % (fea_name, classifier_name)),
    #         dpi=300)
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     plot_confusion_matrix(conf, classes=m_ordered_classes_names, normalize=True,
    #                           title='Conf. matrix %s features %s [b_acc:%.3f]' % (
    #                               fea_name, classifier_name, balanced_accuracy_score(m_labels,
    #                                                                                  m_pred)))
    #     plt.savefig(os.path.join(out_path,
    #                              "conf_matrix_%s_features_%s_normed_merged.png" % (fea_name, classifier_name)), dpi=300)
    #
    #     print("Done!")


