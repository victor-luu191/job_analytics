import numpy as np
import pandas as pd
from scipy.stats import entropy
from numpy.linalg import norm

def klDiv(p, q):
	if any(_q == 0 for _q in q):
		return np.inf

	return sum(_p * np.log2(_p / _q) for _p, _q in zip(p, q) if _p != 0)

def jsDiv(p, q):
	m = np.add(p,q)*0.5
	return 0.5*(klDiv(p, m) + klDiv(q, m))

# entropy() in the function below is actually KL div
def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * np.add(_P, _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def getLowerFences(values):
    q1, q3 = values.quantile(.25), values.quantile(.75); iqr = q3 - q1
    lif, lof = q1 - 1.5*iqr, q1 - 3*iqr
    return lif, lof

def getUpperFences(values):
   	q1, q3 = values.quantile(.25), values.quantile(.75); iqr = q3 - q1
   	uif, uof = q3 + 1.5*iqr, q3 + 3*iqr
   	return uif, uof   