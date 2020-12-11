from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from .class_vectors import predict_by_centroid, predict_by_tf_idf, compute_class_centroids, \
    compute_class_tf_idf, transform_vector_to_tf_idf_class_form, predict_by_cosine, predict_by_euclidean
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


def classify_bopf(X_train, X_test, y_train, class_method, use_pca, scale, n_com, with_mean):
    if class_method == "centroid":
        X_train, y_train = compute_class_centroids(X_train, y_train)
    elif class_method == "tf_idf":
        X_train, y_train = compute_class_tf_idf(X_train, y_train)
        X_test = transform_vector_to_tf_idf_class_form(X_test)

    # else:
    #     std_scaler.fit(X_train)
    #     means = std_scaler.mean_
    #     X_train = X_train - means
    #     X_test = X_test - means

    if use_pca:
        if scale:
            X_train = X_train.todense()
            X_test = X_test.todense()
            std_scaler = StandardScaler(with_mean=with_mean)
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
            pca = PCA(n_components=n_com)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        else:
            svd = TruncatedSVD(n_components=n_com, n_iter=20)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)

    if class_method == "centroid":
        y_pred = predict_by_centroid(X_train, y_train, X_test)
    elif class_method == "tf_idf":
        y_pred = predict_by_tf_idf(X_train, y_train, X_test)
    else:
        raise ValueError("class type '%s' unknown" % class_method)

    return y_pred


def classify_tf_idf(X_train, X_test, y_train, class_method, use_pca, scale, n_com, with_mean, dist_method):
    tfidf_transf = TfidfTransformer()
    X_train = tfidf_transf.fit_transform(X_train)
    X_test = tfidf_transf.transform(X_test)

    if use_pca:
        if scale:
            X_train = X_train.todense()
            X_test = X_test.todense()
            std_scaler = StandardScaler(with_mean=with_mean)
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
            pca = PCA(n_components=n_com)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        else:
            svd = TruncatedSVD(n_components=n_com, n_iter=20)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)

    if class_method == "centroid":
        X_train, y_train = compute_class_centroids(X_train, y_train)

    if dist_method == "euclidean":
        y_pred = predict_by_centroid(X_train, y_train, X_test)
    elif dist_method == "cosine":
        y_pred = predict_by_tf_idf(X_train, y_train, X_test)
    else:
        raise ValueError("dist method '%s' unknown" % dist_method)
    return y_pred


def cv_classify(vectors, labels, use_pca=True, cv_method="loo", class_method="centroid", n_splits=5, scale=True,
                with_mean=True, n_components=20, dist_method="cosine", repr_method="bopf"):
    # vectors_dense = vectors.todense()
    # print("vectors_dense shape: ", vectors.shape)

    if cv_method == "loo":
        cv_splitter = LeaveOneOut()
        cv_iter = cv_splitter.split(vectors)
    else:
        cv_splitter = StratifiedKFold(n_splits=n_splits)
        cv_iter = cv_splitter.split(vectors, labels)

    if class_method is not None:
        n_com = len(np.unique(labels))
    else:
        n_com = min(n_components, vectors.shape[1]-1)

    # if repr_method != "bopf":
    #     n_components = min(vectors.shape[0]-1, n_components)


    real = []
    pred = []
    for train_index, test_index in cv_iter:
        X_train, X_test = vectors[train_index], vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        if repr_method == "bopf":
            y_pred = classify_bopf(X_train, X_test, y_train, class_method, use_pca, scale,
                                      n_com, with_mean)
        else:
            y_pred = classify_tf_idf(X_train, X_test, y_train, class_method, use_pca, scale, n_com,
                                        with_mean, dist_method)

        real.append(y_test)
        pred.append(y_pred)

    return real, pred


def simple_train_test_classify(X_train, X_test, y_train, dist_method="cosine"):
    if dist_method == "euclidean":
        y_pred = predict_by_euclidean(X_train, y_train, X_test)
    elif dist_method == "cosine":
        y_pred = predict_by_cosine(X_train, y_train, X_test)
    else:
        raise ValueError("dist method '%s' unknown" % dist_method)

    return y_pred


def train_test_classify(X_train, X_test, y_train, use_pca=True, class_method="centroid", scale=True,
                        with_mean=True, n_components=20, dist_method="cosine", repr_method="bopf"):
    if repr_method == "bopf":
        if class_method is not None:
            n_com = len(np.unique(y_train))
        else:
            n_com = min(20, X_train.shape[1]-1)
        y_pred = classify_bopf(X_train, X_test, y_train, class_method, use_pca, scale,
                               n_com, with_mean)
    else:
        y_pred = classify_tf_idf(X_train, X_test, y_train, class_method, use_pca, scale, n_components,
                                 with_mean, dist_method)
    return y_pred

    # if class_method is not None:
    #     n_com = len(np.unique(y_train))
    # else:
    #     n_com = min(20, X_train.shape[1]-1)
    #
    # std_scaler = StandardScaler(with_mean=with_mean)
    #
    # if class_method == "centroid":
    #     X_train, y_train = compute_class_centroids(X_train, y_train)
    # elif class_method == "tf_idf":
    #     X_train, y_train = compute_class_tf_idf(X_train, y_train)
    #
    # if use_pca:
    #     if scale:
    #         X_train = X_train.todense()
    #         X_test = X_test.todense()
    #
    #         X_train = std_scaler.fit_transform(X_train)
    #         X_test = std_scaler.transform(X_test)
    #         pca = PCA(n_components=n_com)
    #         dim_before = X_train.shape[1]
    #         X_train = pca.fit_transform(X_train)
    #         X_test = pca.transform(X_test)
    #     else:
    #         svd = TruncatedSVD(n_components=n_com, n_iter=20,   )
    #         dim_before = X_train.shape[1]
    #         X_train = svd.fit_transform(X_train)
    #         X_test = svd.transform(X_test)
    #
    #     print("dim reduced: %d -> %d" % (dim_before, X_train.shape[1]))
    #
    # if class_method == "centroid":
    #     y_pred = predict_by_centroid(X_train, y_train, X_test)
    # elif class_method == "tf_idf":
    #     y_pred = predict_by_tf_idf(X_train, y_train, X_test)
    # else:
    #     raise ValueError("class type '%s' unknown" % class_method)
    #
    # return y_pred