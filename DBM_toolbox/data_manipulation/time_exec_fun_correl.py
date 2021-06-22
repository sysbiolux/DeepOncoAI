# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 02:19:17 2021

@author: sebde
"""


import timeit
sizes =range(10, 200, 10)

setup = '''
import pandas as pd
import numpy as np

N = 
A = pd.DataFrame(np.random.rand(100, N), columns=[str(x) for x in range(N)])
B = pd.DataFrame(np.random.rand(100, N), columns=[str(x) for x in range(N, N*2)])


def test_fun_slow(A, B):
	x = pd.concat([A, B], axis=1).corr().filter(B.columns).filter(A.columns, axis=0)

def test_fun_fast(A, B):
	Az = (A - A.mean())
	Bz = (B - B.mean())
	x = Az.T.dot(Bz).div(len(A)).div(Bz.std(ddof=0)).div(Az.std(ddof=0), axis=0)

'''

results = pd.DataFrame(data=0, index=sizes, columns=['N', 'slow', 'fast'])


for n_features in sizes:

	t_setup = setup
	t_setup = list(t_setup)
	t_setup = t_setup[:45] + [str(n_features)] + t_setup[45:]
	t_setup = "".join(t_setup)

	t_slow = timeit.repeat('test_fun_slow(A, B)', setup=t_setup, repeat = 10, number=1)
	t_fast = timeit.repeat('test_fun_fast(A, B)', setup=t_setup, repeat=10, number=1)

	results.loc[n_features, 'N'] = n_features
	results.loc[n_features, 'slow'] = np.mean(t_slow)
	results.loc[n_features, 'fast'] = np.mean(t_fast)
# 	r = pd.DataFrame(data=np.insert(means, 0, n_features*20).reshape(1,3), index=[str(n_features)], columns=['N', 'slow', 'fast'])
# 	print(r)
# 	results.append(r, ignore_index=True)
# 	print(results)

results.plot(x='N', y=['slow', 'fast'])
