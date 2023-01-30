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

from irm.filter import Filter, IterFilter
from irm.cmp import lt_wrt_u, Counter
from irm.util import sphere, clusters_on_sphere
from experiments.ex import new_ex, get_logger


def my_main(_run, ver, iter, rn, dataset, task, alg,
        n, d, tol, nq, frac
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

    F = None
    if alg.startswith('list'):
        F = Filter
    if alg.startswith('list-lp'):
        F = partial(Filter, solver='LP')
    if F is not None:
        F = partial(IterFilter, F=F, nq=nq, frac=frac)
    ft = F(tol, ult)

    # best
    ubest = -1e7
    for x in D:
        ux = u @ x
        if ux > ubest:
            ubest = ux
    log('ubest', ubest)

    # ... START
    t1_start = process_time()

    for i in range(len(D)):
        x = D[i]

        if ft.prune(x):
            continue
        ft.add(x)

    x = ft.wrapup()
    uret = u @ x
    log('uret', uret)

    t1_stop = process_time()
    # ... END

    log('sz', len(ft))
    log('nft', len(ft.fts))
    log('ncmp', c.count())
    log('ncmperr', cerr.count())
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
        ('alg', ['list']),
        ('task', [task]),

        ('n', [0]),
        ('d', [0]),
        ('tol', [0]),
    ])

    if task.endswith('nq'):
        kv['nq'] = [25,50,100,200,400]
        kv['frac'] = [0.5]
        kv['tol'] = [0.01]
        kv['d'] = [20]
        kv['n'] = [10000]
    if task.endswith('frac'):
        kv['nq'] = [100]
        kv['frac'] = [0.1,0.25,0.5,0.75,0.9]
        kv['tol'] = [0.01]
        kv['d'] = [20]
        kv['n'] = [10000]

    ks, vs = list(kv.keys()), list(kv.values())

    q = []
    for alls in itertools.product(*list(vs)):
        conf = dict([(k, v) for k,v in zip(list(ks),alls)])
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
