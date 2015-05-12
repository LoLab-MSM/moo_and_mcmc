# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# A PyMC implementation of example 2 from MT-DREAM(ZS) original Matlab code.
# 100 dimensional Gaussian distribution

import pymc as pm
import numpy as np
from pymc.backends import text
import pickle
import theano
import theano.tensor as t

def Latin_hypercube(minn, maxn, N):
    y = np.random.rand(N, len(minn))
    x = np.zeros((N, len(minn)))
    
    for j in range(len(minn)):
        idx = np.random.permutation(N)
        P = (idx - y[:,j])/N
        x[:,j] = minn[j] + P * (maxn[j] - minn[j])
    
    return x

log_F = 0

d = 100
A = .5 * np.identity(d) + .5 * np.ones((100,100))
C = np.zeros((100,100))
for i in range(d):
    for j in range(d):
        C[i][j] = A[i][j] * np.sqrt((i+1)*(j+1))

invC = np.linalg.inv(C)

#Create initial samples matrix m that will be loaded in as DREAM history file
m = Latin_hypercube(np.linspace(-5, -5, num=100), np.linspace(15, 15, num=100), 1000)

np.save('ndim_gaussian_seed.npy', m)

@theano.compile.ops.as_op(itypes=[t.dvector], otypes=[t.dscalar])
def likelihood(params):

    log_L = log_F - .5 * np.sum(params*np.dot(invC, params))

    print 'log_L: ',log_L  
    
    return log_L

with pm.Model() as model:
    
    params = pm.Flat('params', shape=(100))       
       
    #log_like = likelihood(model.x, model.y)
    like = pm.ArbLikelihood('like', likelihood(model.params))
    #like = pm.Potential('like', likelihood(model.params))
    
    step = pm.Dream_mpi(blocked=True, start_random=False, save_history=True, history_file='ndim_gaussian_seed.npy', multitry=5, parallel=True)
    
    start = [{'params':m[chain]} for chain in range(3)]
    
    trace = pm.sample(50000, step, start=start, njobs=3, use_mpi=True)
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    text.dump('2015_04_22_ndim_gaussian_mtdreamzs', trace)
    
    pickle.dump(dictionary_to_pickle, open('2015_04_22_ndim_gaussian_mtdreamzs.p', 'wb'))

    
    


