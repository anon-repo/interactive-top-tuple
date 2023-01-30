import numpy as np
from numpy.random import default_rng
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
from irm.util import sphere, clusters_on_sphere
from experiments.ex import new_ex, get_logger


def my_main(_run, ver, iter, rn, dataset, task,
        n, d, tol, nq,
        nsamp
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
    print('u', sum(u), u)
    c, cerr = Counter(), Counter()
    ult = lt_wrt_u(u, c, cerr)

    # best
    ubest = -1e7
    for x in D:
        ux = u @ x
        if ux > ubest:
            ubest = ux
    log('ubest', ubest)

    F = None
    if task.startswith('list'):
        F = partial(Filter, true_u=ubest)
    if task.startswith('list-lp'):
        F = partial(Filter, solver='LP', true_u=ubest)
    if task.startswith('pair'):
        F = partial(PairFilter, true_u=ubest)
    if task.startswith('pair-lp'):
        F = partial(PairFilter, solver='LP', true_u=ubest)
    if task.startswith('hp-lp'):
        F = HyperplaneFilter
    if F is not None:
        F = partial(IterFilter, F=F, nq=nq)

    if task.startswith('rand'):
        F = partial(RandomFilter, nsamp=nsamp)
    if task == 'hp':
        F = BlindHyperplaneFilter
    ft = F(tol, ult)

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

    # random
    Xr = rng.choice(D, size=10)
    log('urand', np.mean([u @ x for x in Xr]))

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
        ('n', [10000]),
        #('n', [1000]),
        ('d', [20]),
        ('tol', [0.1, 0.05, 0.01]),
        ('nq', [100]),

        ('nsamp', [0]),
    ])

    if task.startswith('rand'):
        _, nsp = task.split('-')
        kv['nsamp'] = [int(nsp)]

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
