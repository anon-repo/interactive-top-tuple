import numpy as np
from numpy.random import default_rng
import pandas as pd
import pickle
from time import process_time
from functools import partial
import os
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=4)
import sys
from multiprocessing import Pool
from collections import deque

from irm.filter import Filter, PairFilter, HyperplaneFilter
from irm.cmp import lt_wrt_u, Counter
from irm.util import sphere, clusters_on_sphere
from experiments.ex import new_ex, get_logger


def my_main(_run, ver, iter, rn, dataset, task, alg,
        n, d, tol
            ):
    log = get_logger(_run)
    rng = default_rng(rn)

    u = sphere(1, d, rng).flatten()
    if dataset == 'sphere':
        D = sphere(n, d, rng)
    if dataset.startswith('clusters'):
        ncenter = int(dataset.split('-')[1])
        D = clusters_on_sphere(n, d, ncenter, rng)

    c, cerr = Counter(), Counter()
    ult = lt_wrt_u(u, c, cerr)

    if alg.startswith('list'):
        F = Filter
    if alg.startswith('list-lp'):
        F = partial(Filter, solver='LP')
    if alg == 'pair':
        F = PairFilter
    if alg == 'pair-lp':
        F = partial(PairFilter, solver='LP')
    if alg == 'hp-lp':
        F = HyperplaneFilter
    ft = F(tol, ult)

    q, nq = deque([]), min(100, 10*np.log(len(D)))
    for i in range(len(D)):
        x = D[i]

        if len(q) < nq:
            q.append(x)
            continue

        ft.add(x)
        print(i, len(ft))
        if len(ft) % max(1, i//50) != 0:
            continue

        npruned = sum([1 for y in q if ft.prune(y)])
        if npruned >= nq // 2:
            break

    log('sz', len(ft))
    log('ncmp', c.count())
    log('ncmperr', cerr.count())

    # ... START
    npruned = 0
    t1_start = process_time()
    for x in D:
        if ft.prune(x):
            npruned = npruned + 1

    t1_stop = process_time()
    # ... END

    log('npruned', npruned)
    log('runtime', t1_stop - t1_start)



def run(arg):
    name, conf = arg
    sys.stdout = open(f'./logs/{name}.log', 'w', buffering=1) # 1: line; 0: no buf
    sys.stderr = sys.stdout
    pp.pprint(conf)

    ex = new_ex()
    ex.main(my_main)
    ex.run(config_updates=conf)


if __name__ == '__main__':
    ver = sys.argv[1]
    it = int(sys.argv[2])
    dataset = sys.argv[3]
    task = sys.argv[4]
    is_multiproc = True if sys.argv[5] == 'T' else False

    RNs = [42, 43, 44, 45, 46]
    rn = RNs[it-1]

    kv = dict([
        ('ver', [ver]),
        ('iter', [it]),
        ('rn', [rn]),
        ('dataset', [dataset]),
        #('alg', ['list', 'list-lp', 'list-lb', 'pair-lp', 'hp-lp', 'pair']),
        #('alg', ['pair-lp', 'hp-lp', 'pair']),
        #('alg', ['list', 'list-lp', 'list-lb']),
        #('alg', ['pair']),
        ('alg', ['hp-lp']),
        #('alg', ['pair-lp']),
        ('task', [task]),

        ('n', [0]),
        ('d', [0]),
        ('tol', [0]),
    ])

    if task.endswith('tol'):
        kv['tol'] = [0.1, 0.01, 0.001, 0.0001]
        kv['d'] = [20]
        kv['n'] = [10000]
    if task.endswith('d'):
        kv['tol'] = [0.01]
        kv['d'] = [2, 4, 8, 16, 32, 64, 128]
        #kv['d'] = [256]
        #kv['d'] = [512]
        kv['n'] = [1000]
    if task.endswith('n'):
        kv['tol'] = [0.01]
        kv['d'] = [20]
        kv['n'] = [100,1000,10000,100000]

    ks, vs = list(kv.keys()), list(kv.values())

    q = []
    for alls in itertools.product(*list(vs)):
        conf = dict([(k, v) for k,v in zip(list(ks),alls)])
        name = ''.join([str(v) for v in conf.values()]).replace(' ','').replace(',','-')
        q.append((f'log-{name}', conf))

    if is_multiproc:
        #with Pool(processes=5) as pool:
        #with Pool(processes=3) as pool:
        with Pool(processes=2) as pool:
            pool.map(run, q)
    else:
        for arg in q:
            run(arg)
