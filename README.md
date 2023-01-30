# interactive-top-tuple-code

## Project structure 

Different filters can be found at `irm/filter.py` (interactive regret minimization).

```python
from irm.filter import Filter # the proposed filter, List
from irm.filter import IterFilter # the proposed general iterative framework
from irm.filter import PairFilter, \
                       BlindHyperplaneFilter, \
                       HyperplaneFilter, \
                       RandomFilter # baselines
```

In addition,
* `preproc-data/` includes scripts for data pre-processing. 
* `tests/` includes test cases.
* `experiments/` includes experimental workflows.

Datasets can be downloarded from
* https://www.kaggle.com/datasets/vivovinco/19912021-nba-stats?select=players.csv
* https://www.kaggle.com/datasets/nishiodens/japan-real-estate-transaction-prices
* https://www.kaggle.com/datasets/ekibee/car-sales-information?select=region25_en.csv
* https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_youtube_trending_data.csv, US region, version 800
* https://www.kaggle.com/datasets/fronkongames/steam-games-dataset

Dependent libraries

```
pip install qpsolvers osqp
pip install sacred incense # only needed for experimental workflows
```

## Examples

### Example 1: Create and use a filter

```python
import numpy as np
from numpy.random import default_rng
from functools import partial

from irm.cmp import lt_wrt_u, Counter
from irm.util import sphere
from irm.filter import Filter, IterFilter


# Generate synthetic data
rng = default_rng(42)
tol = 0.01
n, d = 1000, 10
D = sphere(n, d, rng)
u = sphere(1, d, rng).flatten()

c = Counter()
lt = lt_wrt_u(u, c) # less-than operator wrt u

ft = Filter(tol, lt) # List-QP by default 
# ft = Filter(tol, lt, solver='LP') # List-LP

sz = 30 # a fixed sample size
npruned = 0
ubest = -1e7
for i in range(len(D)):
    x = D[i]

    # track best utility
    ux = x @ u
    if ux > ubest:
        ubest = ux

    if len(ft) < sz:
        ft.add(x)

    # prune
    if ft.prune(x):
        npruned = npruned + 1

x = ft.wrapup() # best item kept by the filter
uret = u @ x
        
print(f'prune {npruned} out of {len(D)}')
print(f'#comparisons used = {c.count()}')
print(f'utility: {uret} vs. {ubest}')
```
 
Output
```
prune 947 out of 1000
#comparisons used = 112
utility: 0.5872618427145861 vs. 0.889798019925841
```

### Example 2: Instantiate a filter in the iterative framework

```python
c = Counter()
lt = lt_wrt_u(u, c) # less-than operator wrt u

nq, frac = 25, 5/8 # pool size
F = Filter
# F = partial(Filter, solver='LP')
ft = IterFilter(tol, lt, F=F, nq=nq, frac=frac)

npruned = 0
for i in range(len(D)):
    x = D[i]
    if ft.prune(x):
        npruned = npruned + 1
    else:
        ft.add(x)

x = ft.wrapup() # best item kept by the filter
uret = u @ x

print(f'prune {npruned} out of {len(D)}')
print(f'#comparisons used = {c.count()}')
print(f'utility: {uret} vs. {ubest}')
```

Output
```
prune 864 out of 1000
#comparisons used = 179
utility: 0.889798019925841 vs. 0.889798019925841
```

### Example 3: Allow ties in a comparison

```python
gap = 0.05 # a tie occurs if diff in utility < gap
c, cerr = Counter(), Counter()
ult = lt_wrt_u(u, c, cerr=cerr, gap=gap)
ft = Filter(tol, ult, ties=True)

npruned = 0
for i in range(len(D)):
    x = D[i]

    if len(ft) < sz:
        ft.add(x)
    if ft.prune(x):
        npruned = npruned + 1

x = ft.wrapup() # best item kept by the filter
uret = u @ x
        
print(f'prune {npruned} out of {len(D)}')
print(f'#ties / #comparisons = {cerr.count()} / {c.count()}')
print(f'utility: {uret} vs. {ubest}')
```

Output
```
prune 717 out of 1000
#ties / #comparisons = 19 / 71
utility: 0.5872618427145861 vs. 0.889798019925841
```

## Tests

```
py.test -vv -s
```

## License

MIT license


[//]: # (Comment)



