import pytest
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.optimize import linprog
from functools import cmp_to_key, partial

from irm.filter import Filter, leastsq, PairFilter, BlindHyperplaneFilter, IterFilter
from irm.cmp import lt_wrt_u, binary_search, Counter, cmp_no_ties
from irm.util import sphere


def test_hpfilter():
    u = np.array([1,0])
    D = [
        [1, 0],
        [0, 1],
        [0.1, 0.8],
        [0.8, 0.1],
    ]
    D = np.array(D)

    tol = 0.01
    lt = lt_wrt_u(u)
    ft = BlindHyperplaneFilter(tol, lt)

    for i in range(len(D)):
        x = D[i]

        # Prune
        pruned = ft.prune(x)
        if not pruned:
            ft.add(x)

        assert not pruned or i != 0
        assert not pruned or i != 1
        assert pruned or i != 2
        assert not pruned or i != 3


def test_pairfilter():
    u = np.array([1,0])
    D = [
        [0.9, 0],
        [0.5, 0],
        [0.8, 0],
        [0.5, 0.5],
    ]
    D = np.array(D)

    tol = 0.01
    lt = lt_wrt_u(u)
    ft = PairFilter(tol, lt)

    for i in range(len(D)):
        x = D[i]

        # Prune
        pruned = ft.prune(x)
        if not pruned:
            ft.add(x)

        assert not pruned or i != 0
        assert not pruned or i != 1
        assert pruned or i != 2
        assert not pruned or i != 3


def untest_uncmp():
    '''
    Filter now only keeps one list C for top-1 instaed of two lists for top-1/2.
    Maybe useful in the future.
    '''
    u = np.array([1,0])
    D = [
        [0.9, 0],
        [0.91, 0],
        [0.5, 0],
        [0.51, 0],
        [0.8, 0],
        [0.81, 0],
        [1, 0],
        [1.01, 0],
    ]
    D = np.array(D)

    tol, gap = 0.01, 0.05
    lt = lt_wrt_u(u, gap=gap)
    ft = Filter(tol, lt)

    for i in range(len(D)):
        x = D[i]
        ft.add_with_ties(x)

        s = len(ft)
        c, c0, c1 = len(ft.C), len(ft.C[0]), len(ft.C[1])

        assert c == 2

        assert s == 1 or i != 0
        assert s == 1 or i != 1
        assert c1 > 0 or i != 1

        assert s == 2 or i != 2
        assert s == 2 or i != 3
        assert c0 > 0 or i != 3

        assert s == 3 or i != 4
        assert s == 3 or i != 5
        assert i != 5 or ft.C[0][0] @ u < 0.85
        assert i != 5 or ft.C[1][0] @ u > 0.85

        assert s == 4 or i != 6
        assert s == 4 or i != 7
        assert i != 7 or ft.C[0][0] @ u < 0.95
        assert i != 7 or ft.C[1][0] @ u > 0.95


def test_filter_ties():
    u = np.array([1,0])
    D = [
        [0.9, 0],
        [0.91, 0],
        [0.8, 0],
        [0.81, 0],
        [0.5, 0],
        [0.51, 0],
        [0.91, 0.3],
        [0.91, 0.5],
        [0.91, 0.8],
    ]
    D = np.array(D)

    tol, gap = 0.005, 0.05
    ult = lt_wrt_u(u, gap=gap)
    ft = Filter(tol, ult, ties=True)

    for i in range(len(D)):
        x = D[i]
        if ft.prune(x):
            continue

        ft.add(x)

    assert len(ft.S) == 3
    assert len(ft.Gs) == 3
    Gs = [len(G) for G in ft.Gs.values()]
    assert min(Gs) == 1 # 0.8
    assert max(Gs) == 5 # all 0.91


def test_filter():
    rng = default_rng(42)

    n, d, sz = 1000, 10, 30
    D = sphere(n, d, rng)
    u = sphere(1, d, rng).flatten()

    tol = 0.01
    lt = lt_wrt_u(u)
    ft = Filter(tol, lt)

    npruned = 0
    best, ubest = None, -1e7
    unpruned = []
    for i in range(len(D)):
        x = D[i]

        # test if x is better than best
        ux = x @ u
        if ux > ubest:
            ubest = ux
            best = x

        if len(ft) < sz:
            ft.add(x)

        # Prune
        pruned = False
        if ft.prune(x):
            npruned = npruned + 1
        else:
            unpruned.append(x)

    # Make sure ft contains at least one feasible item
    print('unpruned:', len(unpruned), 'out of', len(D))
    uunp = max([y @ u for y in unpruned] + [-1e7])
    uft = max([y @ u for y in ft]) + tol
    uft = max(uunp, uft)
    assert uft > ubest or np.isclose(uft, ubest)


def test_leastsq():
    A = np.array([[1., ], [1., ], [1., ]])
    b = np.array([1., 1., 0.])
    G = np.ones((1, 1))
    h = np.array([0])
    sol = leastsq(A, b, G=-G, h=h)
    assert np.isclose([0.66666667], sol)

    def lq(S, x):
        s, S, x = len(S), np.array(S), np.array(x)
        R = [S[i - 1] - S[i] for i, _ in enumerate(S) if i > 0]
        R = np.array(R).T
        c = x - S[-1]
        lb = np.zeros(s - 1)  # box ineqs

        est = leastsq(R, c, lb=lb)
        return est, norm(R @ est - c)

    S = [
        [0.8, 0],
        [0.91, 0],
    ]
    x = [0.9, 0]
    est, sol = lq(S, x)
    assert est is not None
    assert sol < 0.05


def test_lp():
    # min_x c^Tx s.t. A_ub x <= b_ub
    c = np.zeros(2)
    A = [
        [1, 0],
        [0, 1],
        [-1, -1],
    ]
    A = np.array(A)
    b = np.array([0, 0, -1])
    res = linprog(c, A_ub=A, b_ub=b, bounds=None, method='highs')

    assert res.status == 2  # infeasible


def test_lt_no_ties():
    A = np.identity(5)
    arr = [r for r in A]
    lt = lt_wrt_u([0, 0.1, 0.3, 0.5, 1])
    assert binary_search(arr, [0,0,0,0,0.6], lt)[0] == 4
    assert binary_search(arr, [0,0,0,0,1.1], lt)[0] == 5
    assert binary_search(arr, [0,0.9,0,0,0], lt)[0] == 1

    c = Counter()
    lt = lt_wrt_u(np.array([1]), c)
    assert lt(np.array([1]), np.array([2]))
    assert lt(np.array([2]), np.array([3]))
    assert c.count() == 2

    ls = [np.array([1]), np.array([4]), np.array([1.5])]
    cmpr = partial(cmp_no_ties, lt=lt)
    ls.sort(key=cmp_to_key(cmpr))
    assert ls[-1] == np.array([4])


def test_binary_search():
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 21]
    assert binary_search(arr, 8)[0] == 4
    assert binary_search(arr, 0.5)[0] == 0
    assert binary_search(arr, 22)[0] == len(arr)
