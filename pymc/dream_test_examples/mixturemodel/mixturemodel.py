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
import scipy.linalg

log_F = np.array([-10.2880, -9.5949])

d = 10
k = 2
log_prior = np.log(np.array([.3333, .6666]))
mu1 = np.linspace(-5, -5, num=d)
mu2 = np.linspace(5, 5, num=d)
mu = np.array([mu1,mu2])
C = np.identity(d)
L = scipy.linalg.cholesky(C, lower=False)
diagL = np.diag(L)
logDetSigma = 2 * np.sum(np.log(diagL))
cov = np.identity(10)*5
mean = np.linspace(0, 0, num=10)
#Create initial samples matrix m that will be loaded in as DREAM history file
m = np.random.multivariate_normal(mean, cov, size=100)

np.save('mixturemodel_seed.npy', m)

@theano.compile.ops.as_op(itypes=[t.dvector], otypes=[t.dscalar])
def likelihood(params):
    log_lh = np.zeros((k))
    for j in range(2):
        log_lh[j] = -.5 * np.sum((params - mu[j,:])**2) + log_F[j]
    maxll = np.max(log_lh)
    post = np.array(np.exp(log_lh - maxll), dtype='float64')
    density = np.sum(post)
    post = post/float(density)
    log_L = np.log(density) + maxll
    print 'params: ',params,'log_L: ',log_L,'log_lh: ',log_lh,'maxll: ',maxll,'post: ',post,'density: ',density
    
    return log_L

with pm.Model() as model:
    
    params = pm.Flat('params', shape=(10))       
       
    #log_like = likelihood(model.x, model.y)
    like = pm.ArbLikelihood('like', likelihood(model.params))
    #like = pm.Potential('like', likelihood(model.params))
    
    step = pm.Dream_mpi(blocked=True, start_random=False, save_history=True, history_file='mixturemodel_seed.npy', multitry=5, parallel=True)
    
    start = [{'params':m[chain]} for chain in range(3)]
    
    trace = pm.sample(50000, step, start=start, njobs=3, use_mpi=True)
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    text.dump('2015_04_22_mixturemodel_mtdreamzs', trace)
    
    pickle.dump(dictionary_to_pickle, open('2015_04_22_mixturemodel_mtdreamzs.p', 'wb'))

    
    


