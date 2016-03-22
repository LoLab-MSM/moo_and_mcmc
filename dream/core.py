# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:15:49 2016

@author: Erin

"""

import numpy as np
import multiprocess as mp
import Dream_shared_vars
from Dream import Dream, DreamPool
from model import Model
import traceback

def run_dream(parameters, likelihood, nchains=5, niterations=50000, start=None, restart=False, verbose=True, **kwargs):

    if restart:
        if start == None:
            raise Exception('Restart run specified but no start positions given.')
        if 'model_name' not in kwargs:
            raise Exception('Restart run specified but no model name to load history and crossover value files from given.')
    
    model = Model(likelihood=likelihood, sampled_parameters=parameters)
    
    if restart:    
        step_instance = Dream(model=model, variables=parameters, history_file=kwargs['model_name']+'_DREAM_chain_history.npy', crossover_file=kwargs['model_name']+'_DREAM_chain_adapted_crossoverprob.npy', verbose=verbose, **kwargs)

    else:
        step_instance = Dream(model=model, variables=parameters, **kwargs)

    pool = _setup_mp_dream_pool(nchains, niterations, step_instance, start_pt=start)
    
    if type(start) is list:
        args = zip([step_instance]*nchains, [niterations]*nchains, start)
    else:
        args = zip([step_instance]*nchains, [niterations]*nchains, [start]*nchains)  

    returned_vals = pool.map(sample_dream, args)
    sampled_params = [val[0] for val in returned_vals]
    log_ps = [val[1] for val in returned_vals]   
    
    return sampled_params, log_ps

def sample_dream(args):

    try: 
        dream_instance = args[0]
        iterations = args[1]
        start = args[2]
        step_fxn = getattr(dream_instance, 'astep')
        sampled_params = np.empty((iterations, dream_instance.total_var_dimension))
        log_ps = np.empty((iterations, 1))
        q0 = start
        naccepts = 0
        naccepts100win = 0
        for iteration in range(iterations):
            if iteration%10 == 0:
                acceptance_rate = float(naccepts)/(iteration+1)
                print('Iteration: ',iteration,' acceptance rate: ',acceptance_rate)
                if iteration%100 == 0:
                    acceptance_rate_100win = float(naccepts100win)/100
                    print('Iteration: ',iteration,' acceptance rate over last 100 iterations: ',acceptance_rate_100win)
                    naccepts100win = 0
            old_params = q0
            sampled_params[iteration], log_ps[iteration] = step_fxn(q0)
            q0 = sampled_params[iteration]   
            if np.any(q0 != old_params):
                naccepts += 1
                naccepts100win += 1
            
    except Exception as e:
        traceback.print_exc()
        print()
        raise e

    return sampled_params, log_ps


def _setup_mp_dream_pool(nchains, niterations, step_instance, start_pt=None):
    
    min_njobs = (2*len(step_instance.DEpairs))+1
    if nchains < min_njobs:
        raise Exception('Dream should be run with at least (2*DEpairs)+1 number of chains.  For current algorithmic settings, set njobs>=%s.' %str(min_njobs))
    if step_instance.history_file != False:
        old_history = np.load(step_instance.history_file)
        len_old_history = len(old_history.flatten())
        nold_history_records = len_old_history/step_instance.total_var_dimension
        step_instance.nseedchains = nold_history_records
        if niterations < step_instance.history_thin:
            arr_dim = ((np.floor(nchains*niterations/step_instance.history_thin)+nchains)*step_instance.total_var_dimension)+len_old_history
        else:
            arr_dim = np.floor((((nchains*niterations)*step_instance.total_var_dimension)/step_instance.history_thin))+len_old_history
    else:
        if niterations < step_instance.history_thin:
            arr_dim = ((np.floor(nchains*niterations/step_instance.history_thin)+nchains)*step_instance.total_var_dimension)+(step_instance.nseedchains*step_instance.total_var_dimension)
        else:
            arr_dim = np.floor(((nchains*niterations*step_instance.total_var_dimension)/step_instance.history_thin))+(step_instance.nseedchains*step_instance.total_var_dimension)
            
    min_nseedchains = 2*len(step_instance.DEpairs)*nchains
    
    if step_instance.nseedchains < min_nseedchains:
        raise Exception('The size of the seeded starting history is insufficient.  Increase nseedchains>=%s.' %str(min_nseedchains))
        
    current_position_dim = nchains*step_instance.total_var_dimension
    history_arr = mp.Array('d', [0]*arr_dim)
    if step_instance.history_file != False:
        history_arr[0:len_old_history] = old_history.flatten()
    nCR = step_instance.nCR
    crossover_setting = step_instance.CR_probabilities
    crossover_probabilities = mp.Array('d', crossover_setting)   
    ncrossover_updates = mp.Array('d', [0]*nCR)
    delta_m = mp.Array('d', [0]*nCR)
    current_position_arr = mp.Array('d', [0]*current_position_dim)
    shared_nchains = mp.Value('i', nchains)
    n = mp.Value('i', 0)
    tf = mp.Value('c', 'F')
    
    if step_instance.crossover_burnin == None:
        step_instance.crossover_burnin = int(np.floor(niterations/10))
        
    if start_pt != None:
        if step_instance.start_random:
            print('Warning: start position provided but random_start set to True.  Overrode random_start value and starting walk at provided start position.')
            step_instance.start_random = False

    p = DreamPool(nchains, initializer=_mp_dream_init, initargs=(history_arr, current_position_arr, shared_nchains, crossover_probabilities, ncrossover_updates, delta_m, n, tf, ))
    
    return p

def _mp_dream_init(arr, cp_arr, nchains, crossover_probs, ncrossover_updates, delta_m, val, switch):
      Dream_shared_vars.history = arr
      Dream_shared_vars.current_positions = cp_arr
      Dream_shared_vars.nchains = nchains
      Dream_shared_vars.cross_probs = crossover_probs
      Dream_shared_vars.ncr_updates = ncrossover_updates
      Dream_shared_vars.delta_m = delta_m
      Dream_shared_vars.count = val
      Dream_shared_vars.history_seeded = switch