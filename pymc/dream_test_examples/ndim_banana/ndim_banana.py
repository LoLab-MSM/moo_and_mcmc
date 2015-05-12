# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# A PyMC implementation of example 1 from MT-DREAM(ZS) original Matlab code.
# n-dimensional banana-shaped Gaussian distribution (example N=2)

import pymc as pm
import numpy as np
from pymc.backends import text
import pickle
import theano
import theano.tensor as t

b = .1
log_F = -4.1405

C = np.identity(2)
C[0][0] = 100
invC = 1/C
invC[np.where(np.isinf(invC))] = 0

#Create initial samples matrix m that will be loaded in as DREAM history file
mu = np.zeros((2))
cov = 10*np.identity(2)
m = np.random.multivariate_normal(mean=mu, cov=cov, size=20)
np.save('ndim_banana_seed.npy', m)

@theano.compile.ops.as_op(itypes=[t.dvector], otypes=[t.dscalar])
def likelihood(params):
    #print t.set_subtensor(params[0], params[0] + .1 * params[1]**2 - 100*.1).eval({params:[5,8]})
    #edited_params = t.set_subtensor(params[1], params[1] + b * params[0]**2 - 100*b)
    new_param2 = params[1] + b * params[0]**2 - 100*b
    #params[1] = params[1] + b * params[0]**2 - 100*b
    new_params = np.array([params[0], new_param2])
    #print t.dot(invC, edited_params).eval()
    #print edited_params*t.dot(invC, edited_params).eval()
    #print t.sum(edited_params*t.dot(invC, edited_params)).eval()
    #log_L = log_F - .5 * t.sum(edited_params*t.dot(invC, edited_params))
    log_L = log_F - .5 * np.sum(new_params*np.dot(invC, new_params))
    #print log_L.eval()
    print 'params: ',params,'edited params: ',new_params,'log_L: ',log_L  
    
    return log_L

with pm.Model() as model:
    
    params = pm.Flat('params', shape=(2))       
       
    #log_like = likelihood(model.x, model.y)
    like = pm.ArbLikelihood('like', likelihood(model.params))
    #like = pm.Potential('like', likelihood(model.params))
    
    step = pm.Dream_mpi(blocked=True, start_random=False, save_history=True, parallel=True, history_file='ndim_banana_seed.npy', multitry=5)
    
    start = [{'x':m[chain][0], 'y':m[chain][1]} for chain in range(3)]
    
    trace = pm.sample(25000, step, start=start, njobs=3, use_mpi=True)
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    text.dump('2015_04_22_ndim_banana_mtdreamzs', trace)
    
    pickle.dump(dictionary_to_pickle, open('2015_04_22_ndim_banana_mtdreamzs.p', 'wb'))

    
    


