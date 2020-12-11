import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import sparse
# import pdb


def compute_class_centroids(vectors, labels):
    classes = np.unique(labels)
    n_classes = len(classes)
    m, n_features = vectors.shape
    assert vectors.shape[0] == len(labels)
    class_vectors = sparse.lil_matrix((n_classes, n_features))
    # print("centroid", vectors.shape, class_vectors.shape, type(vectors))
    class_count = np.zeros(n_classes)
    # sum all
    for k in range(m):
        lbl = np.where(classes == labels[k])[0][0]
        class_vectors[lbl] += vectors[k]
        class_count[lbl] += 1

    # divide by total number to get centroid
    for i in range(n_classes):
        class_vectors[i] = class_vectors[i] / class_count[i]
    return class_vectors, classes


def compute_class_tf_idf(vectors, labels):
    classes = np.unique(labels)
    n_classes = len(classes)
    m, n_features = vectors.shape
    assert vectors.shape[0] == len(labels)
    class_vectors = np.zeros((n_classes, n_features))
    word_class_count = np.zeros(n_features)
    # summ all
    for k in range(m):
        lbl = np.where(classes == labels[k])[0][0]
        class_vectors[lbl] += vectors[k]

    # csr_vectors = vectors.tocsr()
    # word_class_count = (csr_vectors.sign().multiply(csr_vectors.sign())).sum(axis=0)
    word_class_count = np.sum(vectors.toarray().astype(bool).astype(int), axis=0)
    # print("shape word_class_count:", word_class_count.shape, ", n_features:", n_features)
    idf = np.zeros(n_features)
    for i in range(n_features):
        if word_class_count[i] > 0:
            idf[i] = np.log10(1 + n_classes / word_class_count[i])

    for i in range(n_classes):
        for j in range(n_features):
            if class_vectors[i, j] > 0:
                class_vectors[i, j] = (1 + np.log10(class_vectors[i,j])) * idf[j]

    # class_vectors = np.where(np.isinf(class_vectors), 0, class_vectors)
    # class_vectors = np.where(np.isnan(class_vectors), 0, class_vectors)
    return class_vectors, classes


def transform_vector_to_tf_idf_class_form(vector):
    # vec = 1 + np.log10(vector)
    vec = np.zeros(vector.shape)
    if len(vector.shape) == 2:
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1]):
                if vector[i, j] > 0:
                    vec[i, j] = 1 + np.log10(vector[i, j])
    else:
        for i in range(vector.shape):
            if vector[i] > 0:
                vec[i] = 1 + np.log10(vector[i])
    # vec = np.where(np.isinf(vec), 0, vec)
    # vec = np.where(np.isnan(vec), 0, vec)
    return vec


def predict_by_centroid(vectors, labels, test_vectors):
    pred_labels = []
    n, m = test_vectors.shape
    c, v = vectors.shape
    assert m == v
    assert c == len(labels)
    dmatrix = euclidean_distances(test_vectors, vectors)
    for i in range(n):
        min_idx = dmatrix[i].argmin()
        pred_labels.append(labels[min_idx])
    return pred_labels


def predict_by_tf_idf(vectors, labels, test_vectors):
    pred_labels = []
    n, m = test_vectors.shape
    c, v = vectors.shape
    assert m == v
    assert c == len(labels)
    dmatrix = cosine_similarity(test_vectors, vectors)
    for i in range(n):
        max_idx = dmatrix[i].argmax()
        pred_labels.append(labels[max_idx])
    return pred_labels


def predict_by_euclidean(vectors, labels, test_vectors):
    n, m = test_vectors.shape
    c, v = vectors.shape
    assert m == v
    assert c == len(labels)
    pred_labels = []
    dmatrix = euclidean_distances(test_vectors, vectors)
    for i in range(n):
        min_idx = dmatrix[i].argmin()
        pred_labels.append(labels[min_idx])
    return pred_labels


def predict_by_cosine(vectors, labels, test_vectors):
    n, m = test_vectors.shape
    c, v = vectors.shape
    assert m == v
    assert c == len(labels)
    pred_labels = []
    dmatrix = cosine_similarity(test_vectors, vectors)
    # pdb.set_trace()
    for i in range(n):
        max_idx = dmatrix[i].argmax()
        pred_labels.append(labels[max_idx])
    return pred_labels