# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:14:40 2015

@author: Erin
"""

import bayessb
from optparse import OptionParser
import numpy as np
from time import strftime

#Take input command line inputs for random seeds, output names, and MCMC walk lengths (makes starting multiple chains on different CPUs easier).
parser = OptionParser()
parser.add_option('-r', '--random-seed', action='store', type='int', dest='randomseed')
parser.add_option('-o', '--output-file', action='store', type='string', dest='output_file')
parser.add_option('-l', '--length-walk', action='store', type='int', dest='walk_length')
(options, args) = parser.parse_args()
randomseed = options.randomseed
if randomseed == None:
    randomseed = 1
output_file_prefix = options.output_file
if output_file_prefix == None:
    output_file_prefix = ''    
walk_length = options.walk_length
if walk_length == None:
    walk_length = 50000

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

def likelihood(mcmc, params):
    new_param2 = params[1] + b * params[0]**2 - 100*b
    new_params = np.array([params[0], new_param2])
    log_L = log_F - .5 * np.sum(new_params*np.dot(invC, new_params))
    
    try:
        totalerror[mcmc.iter] = log_L
    except AttributeError:
        pass
    return log_L  

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 1 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)
    if mcmc.iter % 1000 == 0:
        np.savetxt(str(iter)+output_file_prefix+'alltestedpositions.txt', mcmc.positions)
        np.savetxt(str(iter)+output_file_prefix+'totalobj.txt', log_L)
        np.savetxt(str(iter)+output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)

#Set BayesSB parameters
opts = bayessb.MCMCOpts()
opts.anneal_length = 0
opts.likelihood_fn = likelihood
opts.step_fn = step
opts.nsteps = walk_length
opts.seed = randomseed
opts.T_init = 1

totalerror = np.zeros(opts.nsteps)

mcmc = bayessb.MCMC(opts)

print 'Starting time:', strftime("%a, %d %b %Y %I:%M:%S")
mcmc.run()
print 'Ending time:', strftime("%a, %d %b %Y %I:%M:%S")

np.savetxt(output_file_prefix+'alltestedpositions.txt', mcmc.positions)
np.savetxt(output_file_prefix+'totalobj.txt', totalerror)
np.savetxt(output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)
								
								