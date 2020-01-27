
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from matplotlib import pyplot as plt

def jsd(p, q, base=np.e):
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p+q)
    return sp.stats.entropy(p,m, base=base)/2. + sp.stats.entropy(q, m, base=base)/2.

x = np.arange(-10,10,0.001)
p = norm.pdf(x, 4.3, 3.5)
q = norm.pdf(x, 4.1, 3.6)
print(p)
print(len(p))
plt.title('jsd1')
print(jsd(p, q))
plt.plot(x, p, c='red')
plt.plot(x, q, c='black')

q = norm.pdf(x, 5, 4)
plt.title('jsd2')
print(jsd(p, q))
plt.plot(x, p, c='red')
plt.plot(x, q, c='black')
