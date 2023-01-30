import numpy as np
from numpy.linalg import norm


class Counter:
    def __init__(self):
        self.cnt = 0

    def inc(self):
        self.cnt = self.cnt + 1

    def count(self):
        return self.cnt


def cmp_no_ties(x, y, lt=None):
    b = lt(x, y)
    if b is None:
        return 0
    return -1 if b else 1


def lt_wrt_u(u, c: Counter=None, cerr: Counter=None, gap: float=None):
    '''
    returns a `lt` function, that compares any (x,y) and returns True|False|None.
    :param u: utility vector
    :return: boolean, or None if (x,y) are uncomparable.
    '''
    def wrapper(x,y):
        if c is not None:
            c.inc()

        uxy = (y - x) @ u
        gxy = np.abs(uxy)
        if gap is not None:
            if gxy < gap or np.isclose(gxy,gap):
                if cerr is not None:
                    cerr.inc()
                return None

        # Handle rounding error
        if np.isclose(uxy, 0):
            if cerr is not None:
                cerr.inc()
            return None

        return uxy > 0

    return wrapper


def binary_search(arr: list, x, lt=None):
    '''
    arr is sorted in ascending order.
    arr must not contain item x, i.e., no equality.
    :param lt: a Callable function
    :return: (i, b)
    - an index i to be inserted for x; i>len(arr) or arr[i] > x
    - b = False if x is uncomparable to the item at i.
    '''
    if lt is None:
        lt = lambda x,y: x < y

    start = 0
    stop = len(arr) - 1
    while start <= stop:
        middle = round((start + stop) / 2)
        guess = arr[middle]

        b = lt(x, guess)
        if b is None:
            return middle, False
        if b:
            stop = middle - 1
        else:
            start = middle + 1

    return start, True
