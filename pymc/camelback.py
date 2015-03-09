# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:21:45 2015

@author: Erin
"""

#PyMC implementation of six-hump camelback function (Dixon, L. C. W. and Szego, G. P.: The optimization problem: An introduction. in Dixon, L. C. W. and Szego, G. P. (Eds.), Towards Global Optimization II, New York: North Holland, 1978.)

import theano
import pymc as pm
import pickle
import theano.tensor as t
import numpy as np

@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar], otypes=[t.dscalar])
def likelihood(x, y):
    term1 = (4- (2.1*(x**2)) + ((x**4)/3))*(x**2)
    term2 = x*y
    term3 = (-4 + (4*(y**2)))*(y**2)
    summed = term1 + term2 + term3
    print 'Cost: ',summed,' x: ',x,' y: ',y
    return np.array(summed, dtype='float64')
    

model = pm.Model()

with model:
    pm.Uniform('x', lower=-3, upper=3, dtype='float64')
    pm.Uniform('y', lower=-2, upper=2, dtype='float64')
    
    cost = likelihood(model.x, model.y)
    
    pm.Normal('globalmin', mu=cost, sd=1e-4, observed=np.array([-1.0316]))
    #pm.Normal('localmin1', mu=cost, sd=1e-4, observed=np.array([-.2155]))

    step = pm.Dream(snooker=0)
    
    trace = pm.sample(800000, step, njobs=5)
    
    dictionary_to_pickle = {}
    
    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('2015_03_09_camelback_dream.p', 'wb'))
    