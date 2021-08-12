from statsmodels.multivariate.manova import MANOVA
import numpy as np
from numpy.linalg import LinAlgError
from scipy import sparse


def manova_rank(X, y, n_variables):
    n_observations, m = X.shape
    n_features = m // n_variables
    assert n_features * n_variables == m
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
