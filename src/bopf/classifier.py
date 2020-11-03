import numpy as np


def classify(test_bop, centroids, tlabel, mt, c, fea_num):
    res = np.zeros(mt, dtype=int)
    for i in range(mt):
        rmin = np.inf
        label = -1
        for j in range(c):
            dist = 0
            for k in range(fea_num):
                q = test_bop[i + k*mt]
                s = centroids[j + k*c]
                dist += (q - s) ** 2
            if dist < rmin:
                rmin = dist
                label = tlabel[j]
        res[i] = label
    return res


def classify2(test_bop, tf_idfs, tlabel, mt, c, fea_num):
    res = np.zeros(mt, dtype=int)
    for i in range(mt):
        rmax = -np.inf
        label = -1
        for j in range(c):
            r1 = 0.0
            r2 = 0.0
            r3 = 0.0
            for k in range(fea_num):
                q = test_bop[i + k * mt]
                if q > 0:
                    q = 1 + np.log10(q)
                s = tf_idfs[j + k * c]
                r1 += q*s
                r2 += q*q
                r3 += s*s

            if r2 != 0 and r3 != 0:
                dist = r1*r1 / (r2*r3)
                if dist > rmax:
                    rmax = dist
                    label = tlabel[j]
        res[i] = label
    return res


# def classify2(test_bop, tf_idfs, tlabel, mt, c, fea_num):
#     res = np.zeros(mt, dtype=int)
#     for i in range(mt):
#         rmax = -np.inf
#         label = -1
#         for j in range(c):
#             r1 = 0.0
#             r2 = 0.0
#             r3 = 0.0
#             pm = 0
#             pv = 0
#             for k in range(fea_num):
#                 pm += test_bop[i + k * mt]
#                 pv += tf_idfs[j + (k) * c]
#                 d = pm
#                 if d > 0:
#                     d = 1 + np.log10(d)
#                 r1 += d * pv
#                 r2 += d * d
#                 r3 += pv * pv
#             r = r1 * r1 / (r2 * r3)
#             if r > rmax:
#                 rmax = r
#                 label = tlabel[j]
#         res[i] = label
#     return res