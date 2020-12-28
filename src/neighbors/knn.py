from sklearn.neighbors import KNeighborsClassifier as skl_knn


class KNeighborsClassifier(skl_knn):

    def __init__(self, n_neighbors=1, classes=None, useClasses=False):
        self.classes = classes
        self.useClasses = useClasses
        super(KNeighborsClassifier, self).__init__(n_neighbors=n_neighbors)

    def fit(self, X, y):
        if self.useClasses:
            return super(KNeighborsClassifier, self).fit(X, self.classes)
        else:
            return super(KNeighborsClassifier, self).fit(X, y)
