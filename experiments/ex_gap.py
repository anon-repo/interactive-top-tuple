import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm
import pandas as pd
import pickle
from time import process_time
from functools import partial, cmp_to_key
import os
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=4)
import sys
from multiprocessing import Pool

from irm.filter import Filter, PairFilter, BlindHyperplaneFilter, IterFilter, HyperplaneFilter, RandomFilter
from irm.cmp import lt_wrt_u, Counter
from irm.util import sphere
from experiments.ex import new_ex, get_logger


def my_main(_run, ver, iter, rn, dataset, task,
        n, d, tol, gap, nq
            ):
    log = get_logger(_run)
    rng = default_rng(rn)

    if dataset == 'sphere':
        D = sphere(n, d, rng)
    else:
        with open(f'datasets/{dataset}.pkl', 'rb') as fin:
            items, names = pickle.load(fin)
        D = items.values
        print('D', D.shape)
        rng.shuffle(D) # in place
    u = sphere(1, D.shape[1], rng).flatten()
    c, cerr = Counter(), Counter()
    ult = lt_wrt_u(u, c, cerr, gap=gap)

    # find best item
    xbest, ubest = None, -1e7
    for x in D:
        ux = u @ x
        if ux > ubest:
            xbest = x
            ubest = ux
    log('ubest', ubest)

    F = None
    if task.startswith('list'):
        F = partial(Filter, ties=True, true_u=ubest)
    if task.startswith('list-lp'):
        F = partial(Filter, solver='LP', ties=True, true_u=ubest)
    if task.startswith('pair-lp'):
        F = partial(PairFilter, solver='LP')
    if task.startswith('hp-lp'):
        F = HyperplaneFilter
    if F is not None:
        F = partial(IterFilter, F=F, nq=nq)
    ft = F(tol, ult)

    # ... START
    t1_start = process_time()

    for i in range(len(D)):
        x = D[i]

        if ft.prune(x):
            continue
        ft.add(x)

    xret = ft.wrapup()

    t1_stop = process_time()
    # ... END

    log('uret', u @ xret)

    log('ncmp', c.count())
    log('ncmperr', cerr.count())
    log('mem', len(ft))
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
        ('task', [task]),
        ('n', [1000]),
        ('d', [20]),
        ('gap', [0.001, 0.01, 0.1]),
        ('nq', [100]),
        ('tol', [0.1]),
    ])

    ks, vs = list(kv.keys()), list(kv.values())
    q = []
    for alls in itertools.product(*list(vs)):
        conf = dict([(k, v) for k,v in zip(list(ks),alls)])

        #conf['tol'] = conf['gap'] / 2

        name = ''.join([str(v) for v in conf.values()]).replace(' ','').replace(',','-')
        q.append((f'log-{name}', conf))

    if is_multiproc:
        #with Pool(processes=5) as pool:
        with Pool(processes=3) as pool:
        #with Pool(processes=2) as pool:
            pool.map(run, q)
    else:
        for arg in q:
            run(arg)
