# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:31:48 2015

@author: Erin
"""

#PyMC script for running MT-DREAM(ZS) test problem #1 from Laloy and Vrugt 2012, section 3.1.
#This test problem is a 200 dimensional multivariate normal distribution.

import theano
import pymc as pm
import pickle
import theano.tensor as t
import numpy as np
from pymc.backends import text

model = pm.Model()

cov_matrix = np.zeros((200,200))
for j1 in range(0, 200):
    for j2 in range(0, 200):
        if j1==j2:
            cov_matrix[j1][j2] = j1+1
        else:
            cov_matrix[j1][j2] = .5

prec_matrix = 1/cov_matrix

data = np.random.multivariate_normal(mean=np.array([0]*200), cov=cov_matrix, size=10000)
data = data.T
sd = np.array([[i for i in range(1,201)]], dtype='float64')
sd = sd.T


with model:
    #The prior is a drawn from a uniform distribution from -5 to 15 for each of the 200 dimensions.
    params = pm.Uniform('params', lower=-5.0, upper=15.0, shape=(200,1), dtype='float64')
    
    #I can't get the multivariate distribution to run for some reason.  I think a bunch of normal distributions with the correct mu and sds should work...
    pm.Normal('true_dist', mu=params, sd=sd, observed=data)
    #pm.MvNormal('true_dist', mu=params, tau=prec_matrix, observed=data)
    
    #Algorithmic values given in paper (eps= b* in paper)
    step = pm.Dream(DEpairs=1, snooker=.1, nseedchains=2000, eps=10e-6, nCR=3, multitry=5)
    
    #njobs = number of chains
    trace = pm.sample(400000, step, njobs=3)
    
    dictionary_to_pickle = {}
    
    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('2015_03_17_mv_test_mtdreamzs.p', 'wb'))     
    
    print 'Saving trace to text...'
    text.dump('2015_03_17_mv_test_mtdreamzs', trace)
    print 'Trace saved successfully.'
    