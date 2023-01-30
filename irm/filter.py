import abc
import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm
from qpsolvers import solve_ls
from scipy.optimize import linprog
from functools import partial, cmp_to_key
from functools import partial
from collections import deque
from itertools import combinations, product

from irm.util import le, gt, myhash
from irm.cmp import binary_search


def leastsq(R, c, **kwargs):
    '''
    https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
    min_x norm(Rx-c) s.t. Gx <= h

    https://scaron.info/doc/qpsolvers/supported-solvers.html#module-qpsolvers.solvers.osqp_
    by default,
    eps_abs, eps_rel = 1e-5
    max_iter = 4000

    :return: optimal objective value
    '''
    return solve_ls(R, c, **kwargs, solver='osqp')


class FilterBase(abc.ABC):
    def __init__(self, tol: float, ult, maxiter: int=4000):
        self.tol = tol
        self.ult = ult
        self.maxiter = maxiter
        self.C = [] # keep uncomparable items

    def find_top(self, X):
        best = None
        for x in X:
            if best is None:
                best = x
            if self.ult(best, x): # drop x if uncomparable
                best = x
        return best

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def add(self, x):
        pass

    def prune_basic(self, x) -> bool:
        for y in self: # __iter__
            if le(norm(x - y), self.tol):
                return True
        return False

    @abc.abstractmethod
    def prune(self, x, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def wrapup(self):
        pass


class Filter(FilterBase):
    '''
    Filter by a cone formed by a sorted list of items
    '''
    def __init__(self, tol: float, ult, solver='QP', ties=False, true_u=1.0):
        '''
        :param solver: 'LP' or 'QP'
        '''
        super().__init__(tol, ult)
        self.solver = solver
        self.ties = ties
        self.true_u = true_u # if true_u < 1, we set tol = tol*true_u in pruning
        self.S = [] # sorted sample/representatives in ascending order
        self.Gs = dict() # maps x to a group

    def add(self, x):
        s = len(self.S)
        idx, is_cmp = binary_search(self.S, x, lt=self.ult)
        if is_cmp:
            self.S.insert(idx, x) # it is ok that idx = len(S)
            if self.ties:
                self.Gs[myhash(x)] = [x]
        else:
            if not self.ties:
                return # discard x

            # insert x into the group with idx
            y = self.S[idx]
            G = self.Gs[myhash(y)]
            G.append(x)

    def __len__(self):
        if self.ties:
            if len(self.Gs) == 0:
                return 0
            return sum([len(G) for G in self.Gs.values()])
        else:
            return len(self.S)

    def __iter__(self):
        if self.ties:
            for G in self.Gs.values():
                yield from G # may have duplicates
        else:
            yield from self.S

    def prune_lp(self, x) -> bool:
        # min_x c^Tx s.t. A_ub x <= b_ub, A_eq x = b_eq
        S, s = self.S, len(self.S)
        Aeq = [S[i-1]-S[i] for i,_ in enumerate(S) if i>0]
        Aeq = np.array(Aeq).T
        beq = x - S[-1]
        c = np.zeros(Aeq.shape[1])

        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs', options={'maxiter': self.maxiter})

        if res.status == 0: # success
            return True
        return False

    def prune_cone(self, x) -> bool:
        S, s = self.S, len(self.S)
        if self.solver == 'LP':
            return self.prune_lp(x)

        # Run constrained least-squares below
        # min_x norm(Rx-c) s.t. Gx <= h
        R = [S[i-1]-S[i] for i,_ in enumerate(S) if i>0]
        R = np.array(R).T
        c = x - S[-1]
        lb = np.zeros(s-1) # box ineqs

        est = leastsq(R, c, lb=lb, max_iter=self.maxiter)

        if est is None: # fails to solve
            return False
        return le(norm(R @ est - c), self.tol*self.true_u)

    def prune_lp_ties(self, x) -> bool:
        # min_x c^Tx s.t. A_ub x <= b_ub, A_eq x = b_eq
        R1 = []
        Gs = [self.Gs[myhash(y)] for y in self.S]
        for G1, G2 in zip(Gs, Gs[2:]):
            for y1,y2 in product(G1,G2):
                R1.append(y1-y2)

        R2 = []
        for idx in [-1,-2]:
            Gbest = self.Gs[myhash(self.S[idx])]
            for y in Gbest:
                R2.append(y)

        if len(R1) == 0:
            R = R2
        else:
            R = np.concatenate([R1,R2])
        R = np.array(R).T

        A = np.array([0]*len(R1) + [1]*len(R2)).reshape((1,-1))
        b = np.array([1])

        A_eq = np.concatenate([R,A])
        b_eq = np.concatenate([x,b])
        c = np.zeros(A_eq.shape[1])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs', options={'maxiter': self.maxiter})

        if res.status == 0: # success
            return True
        return False

    def prune_cone_ties(self, x) -> bool:
        if self.solver == 'LP':
            return self.prune_lp_ties(x)

        # Run constrained least-squares below
        # min_x norm(Rx-c) s.t. Gx <= h and Ax = b
        R = []
        Gs = [self.Gs[myhash(y)] for y in self.S]
        for G1, G2 in zip(Gs, Gs[2:]):
            for y1,y2 in product(G1,G2):
                R.append(y1-y2)
        nR1 = len(R)

        for idx in [-1,-2]:
            Gbest = self.Gs[myhash(self.S[idx])]
            for y in Gbest:
                R.append(y)
        nR = len(R)
        R = np.array(R).T
        c = x
        lb = np.zeros(nR) # box ineqs

        A = np.array([0]*nR1 + [1]*(nR-nR1))
        b = np.array([1])
        est = leastsq(R, c, A=A, b=b, lb=lb, max_iter=self.maxiter)

        if est is None: # fails to solve
            return False
        return le(norm(R @ est - c), self.tol * self.true_u)

    def prune(self, x) -> bool:
        if self.prune_basic(x):
            return True
        if len(self.S) < 2:
            return False
        if self.ties:
            return self.prune_cone_ties(x)
        return self.prune_cone(x)

    def wrapup(self):
        return self.S[-1]


class IterFilter(FilterBase):
    def __init__(self, tol: float, ult, F: FilterBase, nq: int=100, frac=0.5):
        super().__init__(tol, ult)
        self.F = partial(F, tol=tol, ult=ult)
        self.fts = []
        self.ft = self.F()
        self.q = []
        self.q_ed = set()
        self.nq = nq
        self.frac = frac

    def __len__(self):
        return sum([len(ft) for ft in self.fts])

    def __iter__(self):
        pass

    def prune(self, x) -> bool:
        for _ft in self.fts:
            if _ft.prune(x):
                return True
        return False

    def add(self, x):
        if len(self.q) < self.nq:
            self.q.append(x)
            return

        self.ft.add(x)
        if len(self.ft) % max(1, len(self.ft)//50) != 0: # try less frequently for large ft
            return

        pruned = [i for i,y in enumerate(self.q) if i not in self.q_ed and self.ft.prune(y)]
        self.q_ed.update(pruned)
        if len(self.q_ed) >= self.nq * self.frac:
            self.fts.append(self.ft)
            self.ft = self.F()
            self.q = [x for i, x in enumerate(self.q) if i not in self.q_ed]
            self.q_ed = set()

    def wrapup(self):
        # Add all items in pool to the last filter
        for x in self.q:
            if not self.ft.prune(x):
                self.ft.add(x)
        if len(self.ft) > 0:
            self.fts.append(self.ft)
            self.ft = self.F()

        # Sort top-1 items, one in each ft
        X = [ft.wrapup() for ft in self.fts]
        return self.find_top(X)


class PairFilter(FilterBase):
    '''
    Filter by a cone formed by pairs of items
    '''
    def __init__(self, tol: float, ult, solver='QP', true_u=1.0):
        super().__init__(tol, ult)
        self.solver = solver
        self.true_u = true_u # if true_u < 1, we set tol = tol*true_u in pruning
        self.P = [] # list of pairs, given as (x,y) s.t. x < y
        self.p = [] # current pair

    def add(self, x):
        self.p.append(x)
        if len(self.p) == 2:
            x, y = self.p
            b = self.ult(x,y)
            if b is None: # ties, keep either one
                self.C.append(x)
            else:
                self.P.append((x,y) if b else (y,x))
            self.p = []

    def __len__(self):
        return len(self.P) * 2 + len(self.p) + len(self.C)

    def __iter__(self):
        for x,y in self.P:
            yield x
            yield y
        yield from self.C

    def prune_lp(self, x) -> bool:
        # min_x c^Tx s.t. A_ub x <= b_ub, A_eq x = b_eq
        P, s = self.P, len(self.P)
        R1 = [y-z for (y,z) in P]
        R2 = [z for (y,z) in P]
        R = np.concatenate([R1,R2])
        R = np.array(R).T

        A = np.array([0]*s + [1]*s).reshape((1,-1))
        b = np.array([1])

        A_eq = np.concatenate([R,A])
        b_eq = np.concatenate([x,b])
        c = np.zeros(A_eq.shape[1])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs', options={'maxiter': self.maxiter})

        if res.status == 0: # success
            return True
        return False

    def prune(self, x) -> bool:
        if self.prune_basic(x):
            return True

        P, s = self.P, len(self.P)
        if s == 0:
            return False

        if self.solver == 'LP':
            return self.prune_lp(x)

        # Run constrained least-squares below
        # min_x norm(Rx-c) s.t. Gx <= h and Ax = b
        R1 = [y-z for (y,z) in P]
        R2 = [z for (y,z) in P]
        R = np.concatenate([R1,R2])
        R = np.array(R).T
        c = x
        lb = np.zeros(2*s) # box ineqs

        A = np.array([0]*s + [1]*s)
        b = np.array([1])
        est = leastsq(R, c, A=A, b=b, lb=lb, max_iter=self.maxiter)

        if est is None: # fails to solve
            return False
        return le(norm(R @ est - c), self.tol * self.true_u)

    def wrapup(self):
        X = [z for (y,z) in self.P]
        X.extend(self.C)
        return self.find_top(X)


class BlindHyperplaneFilter(PairFilter):
    '''
    Filter by hyperplanes, each formed by a pair of items.
    It doesn't not guarantee to find a feasible item.
    '''
    def __init__(self, tol, ult):
        super().__init__(tol=tol, ult=ult)

    def prune(self, x) -> bool:
        '''
        A hyperplane by pair (x,y) s.t. x < y is as follows.
        u^T(x-y) < 0, so any item z s.t. z^T(x-y) > 0 is pruned.
        '''
        if self.prune_basic(x):
            return True

        P = self.P
        if len(P) == 0:
            return False

        a = [gt(x @ (y-z), 0) for (y,z) in P]
        return np.any(a)


class HyperplaneFilter(PairFilter):
    '''
    Xie, Min, Raymond Chi-Wing Wong, and Ashwin Lall.
    "Strongly truthful interactive regret minimization."
    In Proceedings of the 2019 International Conference on Management of Data, pp. 281-298. 2019.

    For every new item x, run the following LP:
    min_u constant
    s.t.
    u^T(x-z) >= eps for all (y,z) in P,
    u^T(z-y) >= 0 for all (y,z) in P.
    If the LP is infeasible, it means that x is worse than at least one item in P.
    '''
    def __init__(self, tol, ult):
        super().__init__(tol=tol, ult=ult)

    def prune(self, x) -> bool:
        if self.prune_basic(x):
            return True

        if len(self.P) == 0:
            return False

        # min_x c^Tx s.t. A_ub x <= b_ub
        d = len(self.P[0][0])
        c = np.zeros(d)
        A = [z-y for y,z in self.P]
        A.extend([(1-self.tol)*x - z for y,z in self.P])

        A = -np.array(A)
        b1 = np.zeros(len(self.P))
        b2 = -np.ones(len(self.P))
        b = np.concatenate([b1,b2])

        res = linprog(c, A_ub=A, b_ub=b, bounds=(None,None), method='highs', options={'maxiter': self.maxiter})

        if res.status == 0: # success
            return False
        if res.status == 2: # infeasible
            return True
        return False # LP fails


class RandomFilter(FilterBase):
    def __init__(self, tol: float, ult, nsamp: int=30):
        super().__init__(tol, ult)
        self.S = []
        self.nsamp = nsamp

    def __len__(self):
        return len(self.S)

    def __iter__(self):
        yield from self.S

    def prune(self, x) -> bool:
        if self.prune_basic(x):
            return True

        return False

    def add(self, x):
        if len(self.S) >= self.nsamp:
            return

        self.S.append(x)

    def wrapup(self):
        return self.find_top(self.S)
