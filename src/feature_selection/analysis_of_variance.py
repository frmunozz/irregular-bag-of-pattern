from statsmodels.multivariate.manova import MANOVA
import numpy as np
from numpy.linalg import LinAlgError
from scipy import sparse
from numba import jit
import time


@jit(nopython=True)
def feature_grouping_fast(X, n_features, n_variables):
    """THIS FUNCTION HAS A SMALL IMPROVEMENT COMPARED TO THE NORMAL ONE"""
    fea_ranking = []
    for i in range(n_features):

        # get for each feature the corresponding variables based on the flattened matrix
        fea = X[:, n_variables * i]
        for j in range(1, n_variables):
            fea_j = X[:, n_variables * i + j]
            fea = np.hstack((fea, fea_j))
        fea_ranking.append(fea)
    return fea_ranking


def feature_grouping_slow(X, n_features, n_variables):
    """ CREATED FOR COMPARISON WITH THE FAST VERSION"""
    fea_ranking = np.full(n_features, None)
    for i in range(n_features):

        # get for each feature the corresponding variables based on the flattened matrix
        fea = None
        for j in range(n_variables):
            fea_j = X[:, n_variables * i + j]
            if fea is None:
                fea = fea_j
            else:
                fea = np.hstack((fea, fea_j))
        fea_ranking[i] = fea
    return fea_ranking


def manova_rank_fast(X, y, n_variables):
    """THIS FUNCTION HAS A SMALL IMPROVEMENT COMPARED TO THE NORMAL ONE"""
    n_observations, m = X.shape
    n_features = m // n_variables
    fea_ranking = feature_grouping_fast(X.toarray(), n_features, n_variables)
    fea_values = np.zeros(n_features)
    fea_p = np.zeros(n_features)
    count_faileds = 0
    for i in range(n_features):
        try:
            manova = MANOVA(endog=fea_ranking[i], exog=y)
            test = manova.mv_test()
            wilks = test.results["x0"]["stat"].loc["Pillai's trace"]
        except LinAlgError:
            # in this case, the data has perfect correlation and the numerical implemented method will fail due
            # to a singular matrix. Here we will assume a wilks lambda of 1 with Pr > F of 1
            fea_values[i] = 0
            fea_p[i] = 1
            count_faileds += 1
        except Exception as e:
            # print("code fail for feature %d with error: %s" % (i, e))
            # print("Discarding this feature from the process (set wilks lambda to 1)")
            fea_values[i] = 0
            fea_p[i] = 1
            count_faileds += 1
        else:
            fea_values[i] = wilks.Value
            fea_p[i] = wilks["Pr > F"]

    if count_faileds > m // 2:
        print("%d/%d features failed because of absence of dependent variables" % (count_faileds, m))

    return fea_values, fea_p


def manova_rank(X, y, n_variables):
    n_observations, m = X.shape
    n_features = m // n_variables
    # assert n_features * n_variables == m
    assert n_observations == len(y)

    fea_values = np.zeros(n_features)
    fea_p = np.zeros(n_features)
    for i in range(n_features):

        # get for each feature the corresponding variables based on the flattened matrix
        fea = None
        for j in range(n_variables):
            fea_j = X[:, n_variables * i + j]
            if fea is None:
                fea = fea_j
            else:
                fea = sparse.hstack((fea, fea_j), format="csr")
        fea = fea.toarray()
        # print("endog:", fea.shape, type(fea))
        # print("exog:", y.shape, type(y))
        # we try to perform manova.
        try:
            manova = MANOVA(endog=fea, exog=y)
            test = manova.mv_test()
            wilks = test.results["x0"]["stat"].loc["Pillai's trace"]
        except LinAlgError:
            # in this case, the data has perfect correlation and the numerical implemented method will fail due
            # to a singular matrix. Here we will assume a wilks lambda of 1 with Pr > F of 1
            fea_values[i] = 0
            fea_p[i] = 1
        except Exception as e:
            print("code fail for feature %d with error: %s" % (i, e))
            print("Discarding this feature from the process (set wilks lambda to 1)")
            fea_values[i] = 0
            fea_p[i] = 1
        else:
            fea_values[i] = wilks.Value
            fea_p[i] = wilks["Pr > F"]

    # since wilks lambda is inverse (the lower the value the better), and lives in the range (0, 1)
    # we will inverse the values
    # fea_values = 1 - fea_values
    # with this, the current selectKTop from sklearn can be used.

    return fea_values, fea_p


if __name__ == '__main__':
    a=1000
    b=60000
    X = sparse.random(a, b, format="csr")
    X = X.toarray().astype(np.float64)
    print(type(X))
    n_variables = 6
    n_features = b // n_variables
    y = np.random.randint(0, high=10, size=1000)

    print("starting")
    ini = time.time()
    c = feature_grouping_slow(X, n_features, n_variables)
    end = time.time()
    print("TIME FOR MANOVA WITHOUT USING NUMBA: ", end - ini)
    time.sleep(2)
    ini = time.time()
    a = feature_grouping_fast(X, n_features, n_variables)
    end = time.time()
    print("TIME FOR MANOVA USING NUMBA: ", end - ini)
    time.sleep(2)
    ini = time.time()
    b = feature_grouping_fast(X, n_features, n_variables)
    end = time.time()
    print("TIME FOR MANOVA USING NUMBA: ", end - ini)
    print(np.sum(c[0]), np.sum(a[0]), np.sum(b[0]))
    time.sleep(2)
