import numpy as np
from numpy.linalg import norm
from itertools import combinations


def ge(a, b):
    return np.isclose(a,b) or a > b


def le(a, b):
    return np.isclose(a,b) or a < b


def gt(a, b):
    return not np.isclose(a,b) and a > b


def lt(a, b):
    return not np.isclose(a,b) and a < b


# hash is process-unstable; set PYTHONHASHSEED=0
# https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
myhash = lambda x: hash(x.data.tobytes())


def sphere(n, d, rng):
    '''
    n independent random points on the surface of d-dim sphere
    :return: n-by-d matrix
    '''
    D = rng.normal(size=(n, d))
    return D / norm(D, axis=1)[:, None]


def clusters_on_sphere(n, d, ncenter, rng):
    '''
    Independent random centers on the surface of d-dim sphere.
    Then for each center c,
    generate n//ncenter random Guassion points around c within the sphere
    :return: n-by-d matrix
    '''
    C = rng.normal(size=(ncenter, d))
    C = C / norm(C, axis=1)[:, None]
    Ds = []
    for i, c in enumerate(C):
        Dc = rng.normal(loc=c, scale=0.1, size=(n//ncenter*2, d))
        Dc = [x for x in Dc if norm(x) <= 1]
        if len(Dc) > 0:
            Ds.append(Dc[:n//ncenter])

    n1 = n - sum([len(Dc) for Dc in Ds])
    if n1 > 0:
        O = rng.normal(size=(n1, d)) # outliers
        O = O / norm(O, axis=1)[:, None]
        Ds.append(O)

    D = np.concatenate(Ds)
    print(D.shape)
    rng.shuffle(D)
    return D
