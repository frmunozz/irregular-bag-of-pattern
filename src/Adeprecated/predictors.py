from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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