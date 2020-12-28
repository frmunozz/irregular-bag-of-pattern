from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from .class_vectors import compute_class_centroids, compute_class_tf_idf
from .classify import simple_train_test_classify
from sklearn.metrics import balanced_accuracy_score


def cv_fea_num_finder(vectors, labels, n_splits=5, random_state=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    real_labels = []
    pred_centroid = []
    pred_tf_idf = []

    for train_index, test_index in kfold.split(vectors[0], labels):
        X_train1, X_test1 = vectors[0][train_index], vectors[0][test_index]
        X_train2, X_test2 = vectors[1][train_index], vectors[1][test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train1, y_train1 = compute_class_centroids(X_train1, y_train)
        X_train2, y_train2 = compute_class_tf_idf(X_train2, y_train)
        y_pred1 = simple_train_test_classify(X_train1, X_test1, y_train1, dist_method="euclidean")
        y_pred2 = simple_train_test_classify(X_train2, X_test2, y_train2, dist_method="cosine")

        pred_centroid.extend(list(y_pred1))
        pred_tf_idf.extend(list(y_pred2))
        real_labels.extend(list(y_test))

    return balanced_accuracy_score(real_labels, pred_centroid), \
           balanced_accuracy_score(real_labels, pred_tf_idf)
