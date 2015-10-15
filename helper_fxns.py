# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 16:20:59 2014

@author: Erin
"""

import numpy as np
import re
from pymc.stats import hpd
import matplotlib.pyplot as plt
import math
import pickle
from pysb.bng import generate_equations
import os
import dill

def gelman_rubin_trace_dict(trace_dict, burnin=0):
    Rhat = {}
    def calc_rhat(var_dict):
        # a ValueError that will handle the multidimensional case
        n = burnin
            
        last_samples = [var_dict[chain][n::] for chain in range(len(var_dict))]
        
        n = len(var_dict[0])-burnin
        
        # Calculate between-chain variance
        B = n * np.var(np.mean(last_samples, axis=1), ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(last_samples, axis=1, ddof=1))

        # Estimate of marginal posterior variance
        Vhat = W*(n - 1)/n + B/n
    
        return np.sqrt(Vhat/W)
        
    for var in trace_dict:
        Rhat[var] = calc_rhat(trace_dict[var])
    
    return Rhat
    
def create_trace_matrix(trace_dict, burnin=10000, thin=10, chain_num='all'):
    if 'param_list' not in trace_dict:
        trace_dict['param_list'] = [key for key in trace_dict.keys()]
    total_param_dim = 0
    for key in trace_dict:
        try:
            total_param_dim += len(trace_dict[key][0][0])
        except TypeError:
            total_param_dim += 1

    if chain_num == 'all':
        trace_arr = np.zeros(((((len(trace_dict[trace_dict.keys()[0]][0])-burnin)*len(trace_dict[trace_dict.keys()[0]]))/thin), total_param_dim))
    else:
        trace_arr = np.zeros(((len(trace_dict[trace_dict.keys()[0]][chain_num])-burnin)/thin, total_param_dim))
        
    for i, key in enumerate(trace_dict['param_list']):
        if chain_num == 'all':
            chain_list = [trace_dict[key][j][burnin::thin] for j in range(len(trace_dict[key]))]
            try:
                if len(chain_list[0][0]) > 1:
                    for param in range(len(chain_list[0][0])):
                        chain_list_for_param = [chain_list[nchain][:,param] for nchain in range(len(chain_list))]
                        trace_arr[:,param] = np.concatenate(chain_list_for_param)
            except TypeError:
                trace_arr[:,i] = np.concatenate(chain_list)
        else:
            trace_arr[:,i] = trace_dict[key][chain_num][burnin::thin]
    
    return trace_arr
    
def find_most_probable_vals(trace_array, trace_dict, axis=0):
    map_vals = {}
    u, indices = np.unique(trace_array, return_inverse=True)
    map_vector = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(trace_array.shape), None, np.max(indices) + 1), axis=axis)]
    for i, key in enumerate(trace_dict['param_list']):
        map_vals[key] = map_vector[i]
    
    return map_vals
    
def sub_parameters(model, param_dict, log=True, KDs=True, generic_kf=1.5e4):
    generic_kf_log = np.log10(generic_kf)
    if KDs == True:
       for param, value in param_dict.items():
           if 'KD' in param:
               x = re.compile('(KD)(\S*)', re.IGNORECASE)
               y = x.search(param)
               param_name = y.groups()[1]
               kf_param_name = 'kf'+param_name
               kr_param_name = 'kr'+param_name
               if log == True:
                   model.parameters[kf_param_name].value = 10**generic_kf_log
                   print 'Changed parameter '+str(kf_param_name)+' to '+str(10**generic_kf_log)
                   model.parameters[kr_param_name].value = 10**(value+generic_kf_log)
                   print 'Changed parameter '+str(kr_param_name)+' to '+str(10**(value+generic_kf_log))
               else:
                   model.parameters[kf_param_name].value = generic_kf
                   print 'Changed parameter '+str(kf_param_name)+'to '+str(generic_kf)
                   model.parameters[kr_param_name].value = value*generic_kf
                   print 'Changed parameter '+str(kr_param_name)+' to '+str(value*generic_kf)
           else:
               if log == True:
                   model.parameters[param].value = 10**value
                   print 'Changed parameter '+str(param)+' to '+str(10**value)
               else:
                   model.parameters[param].value = value
                   print 'Changed parameter '+str(param)+' to '+str(value)
    else:
        for param, value in param_dict.items():
            if log == True:
                model.parameters[param].value = 10**value
                print 'Changed parameter '+str(param)+' to '+str(10**value)
            else:
                model.parameters[param].value = value
                print 'Changed parameter '+str(param)+' to '+str(value)
                
def check_thermoboxes(param_dict, log=True):
    thermo_dict = {}
    if log == True:
        box1 = (1/(10**param_dict['KD_AA_cat1']))*(1/(10**param_dict['KD_AA_allo2']))*(10**param_dict['KD_AA_cat3'])*(10**param_dict['KD_AA_allo1'])
        box2 = (1/(10**param_dict['KD_AA_allo1']))*(1/(10**param_dict['KD_AG_cat3']))*(10**param_dict['KD_AA_allo3'])*(10**param_dict['KD_AG_cat1'])
        box3 = (1/(10**param_dict['KD_AG_allo1']))*(1/(10**param_dict['KD_AA_cat2']))*(10**param_dict['KD_AG_allo2'])*(10**param_dict['KD_AA_cat1'])
        box4 = (1/(10**param_dict['KD_AG_cat1']))*(1/(10**param_dict['KD_AG_allo3']))*(10**param_dict['KD_AG_cat2'])*(10**param_dict['KD_AG_allo1'])
    
    else:
        box1 = (1/(param_dict['KD_AA_cat1']))*(1/(param_dict['KD_AA_allo2']))*(param_dict['KD_AA_cat3'])*(param_dict['KD_AA_allo1'])
        box2 = (1/(param_dict['KD_AA_allo1']))*(1/(param_dict['KD_AG_cat3']))*(param_dict['KD_AA_allo3'])*(param_dict['KD_AG_cat1'])
        box3 = (1/(param_dict['KD_AG_allo1']))*(1/(param_dict['KD_AA_cat2']))*(param_dict['KD_AG_allo2'])*(param_dict['KD_AA_cat1'])
        box4 = (1/(param_dict['KD_AG_cat1']))*(1/(param_dict['KD_AG_allo3']))*(param_dict['KD_AG_cat2'])*(param_dict['KD_AG_allo1'])     
        
    thermo_dict['box1'] = box1
    thermo_dict['box2'] = box2
    thermo_dict['box3'] = box3
    thermo_dict['box4'] = box4
    
    return thermo_dict

def calc_credible_intervals(trace_arr, trace_dict, alpha=.05):
    hpd_dict = {}
    
    for i, key in enumerate(trace_dict['param_list']):
        hpd_dict[key] = hpd(trace_arr[:,i], alpha)
        
    return hpd_dict

def credible_interval_array(hpd_dict):
    hpd_array = np.zeros((len(hpd_dict), len(hpd_dict[hpd_dict.keys()[0]])))
    i = 0
    for var in hpd_dict.keys():
        hpd_array[i] = hpd_dict[var]
        i += 1

    return hpd_array
    
def plot_histograms(trace_dict, trace_arr, bins=10, plot_original_vals=False, model=None, param_name_change_dict=None, max_yval = .5, title_fontsize=20):
    if plot_original_vals==True and model==None:
        raise Exception('Model needs to be specified')
    
    fig_list = []
    
    #fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    #fig2, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3, 2)
    
    n = 0
    fig, axarr = plt.subplots(3, 2)
    axarr = axarr.flatten()
    xlow = np.min(trace_arr)
    xhigh = np.max(trace_arr)
    for var_num, variable in enumerate(trace_dict['param_list']):
        weights = np.ones_like(trace_arr[:,var_num])/len(trace_arr[:,var_num])
        axarr[n].hist(trace_arr[:,var_num], histtype='stepfilled', bins=bins, color='k', weights=weights)
        if plot_original_vals==True:
            axarr[n].vlines(np.log10(model.parameters[variable].value), 0, .5)
        if param_name_change_dict is not None:
            axarr[n].set_title(str(param_name_change_dict[variable]), fontsize=title_fontsize)
        else:
            axarr[n].set_title(str(variable), fontsize=title_fontsize)
        axarr[n].set_xlabel('Log(parameter value)')
        axarr[n].set_ylabel('Probability')
        axarr[n].set_xlim(np.ceil(xlow-1), np.ceil(xhigh))
        axarr[n].set_ylim(0, max_yval)
        n += 1
        if n == 6:
            fig.subplots_adjust(wspace=.5, hspace=1)
            fig_list.append(fig)
            fig, axarr = plt.subplots(3, 2)
            axarr = axarr.flatten()
            n = 0
        
    #fig1.subplots_adjust(wspace=.5, hspace=1)
    #fig2.subplots_adjust(wspace=.5, hspace=1)

    return fig_list
    
def sample_plots(trace_dict):
    if 'param_list' not in trace_dict:
        trace_dict['param_list'] = [key for key in trace_dict.keys()]
    fig_list = []
    
    n = 0
    nkeys = len(trace_dict.keys())
    fig, axarr = plt.subplots(3, 2)
    axarr = axarr.flatten()
    iterations = range(len(trace_dict[trace_dict.keys()[0]][0]))
    print len(iterations)
    print len(trace_dict[trace_dict.keys()[0]])
    print len(trace_dict[trace_dict.keys()[0]][0])

    for variable in trace_dict['param_list']:
        for chain in range(len(trace_dict[trace_dict.keys()[0]])):
            axarr[n].plot(iterations, trace_dict[variable][chain])
        axarr[n].set_title(str(variable))
        axarr[n].set_xlabel('Iterations')
        axarr[n].set_ylabel('Sampled value')
        axarr[n].set_xlim(0, len(iterations))
        #axarr[n].set_ylim(0, .22)
        n += 1
        if n == 6 or n==nkeys:
            fig.subplots_adjust(wspace=.5, hspace=1)
            fig_list.append(fig)
            fig, axarr = plt.subplots(3, 2)
            axarr = axarr.flatten()
            n = 0
        
    #fig1.subplots_adjust(wspace=.5, hspace=1)
    #fig2.subplots_adjust(wspace=.5, hspace=1)

    return fig_list

def create_ratio_trace_dict(trace_dict):
    trace_ratio_dict = {}
    for param_num_1, parameter_1 in enumerate(trace_dict['param_list']):
        for param_num_2, parameter_2 in enumerate(trace_dict['param_list'][param_num_1+1::]):
            key_name = 'ratio_'+str(parameter_1)+'_'+str(parameter_2)
            trace_ratio_dict[key_name] = [trace_dict[parameter_1][i]/trace_dict[parameter_2][i] for i in range(len(trace_dict[parameter_1]))]
    
    return trace_ratio_dict
    
def convert_param_vec_dict_to_param_dict(param_vec_dict, pysb_parameter_list):
    param_dict = {}
    for param_num, param in enumerate(pysb_parameter_list):
        param_dict[param.name] = [param_vec_dict[param_vec_dict.keys()[0]][chain][:,param_num] for chain in range(len(param_vec_dict[param_vec_dict.keys()[0]]))]
    param_dict['param_list'] = [param.name for param in pysb_parameter_list]
    return param_dict
    
def plot_tsne_data(unique_tsne_output, original_trace_arr, histogram=True, hexbin_gridsize=None):
    if histogram == True:
        original_trace_arr = np.ascontiguousarray(original_trace_arr)
        unique_vecs, inv_idx = np.unique(original_trace_arr.view([('', original_trace_arr.dtype)]*original_trace_arr.shape[1]), return_inverse=True)
        full_tsne_output = unique_tsne_output[inv_idx]
        if hexbin_gridsize != None:
            fig = plt.hexbin(full_tsne_output[:,0], full_tsne_output[:,1], gridsize=hexbin_gridsize)
        else:
            fig = plt.hexbin(full_tsne_output[:,0], full_tsne_output[:,1])
    else:
        fig = plt.scatter(unique_tsne_output[:,0], unique_tsne_output[:,1])
    
    return fig

def create_model_files(model, model_name, directory=None):
    curr_dir = os.getcwd()
    if not model.odes:
        generate_equations(model)
    
    if directory != None:
        os.chdir(directory)
    
    file_basename = model_name+'_model'
    dill.dump(model, open(file_basename+'.p', 'wb'))
    
    os.chdir(curr_dir)

def load_model_files(model_name, directory=None):
    curr_dir = os.getcwd()
    if directory != None:
        os.chdir(directory)
    
    file_basename = model_name+'_model'
    model = dill.load(open(file_basename+'.p'))
    
    os.chdir(curr_dir)
    
    return model
            
def check_gr_over_time(trace_dict, model=None, interval=50000, params_to_remove=None):
    if params_to_remove != None:
        for param in params_to_remove:
            del trace_dict[param]
     
    if model:
        param_vec_dict = convert_param_vec_dict_to_param_dict(trace_dict, model.parameters_rules())
    else:
        param_vec_dict = trace_dict
    gr_results = []
    for end in range(interval, len(trace_dict[trace_dict.keys()[0]][0])+1, interval):
        shorter_dict = {}
        for key in param_vec_dict:
            shorter_trace = [param_vec_dict[key][chain_num][:end] for chain_num in range(len(param_vec_dict[key]))]
            shorter_dict[key] = shorter_trace
        gr = gelman_rubin_trace_dict(shorter_dict)
        params1_2 = 0
        params1_1 = 0
        for param in gr.keys():
            if gr[param] < 1.2:
                params1_2 += 1
            if gr[param] < 1.1:
                params1_1 += 1
    
        
        perc_12 = (float(params1_2)/len(param_vec_dict))*100
        perc_11 = (float(params1_1)/len(param_vec_dict))*100
        results = [params1_2, perc_12, params1_1, perc_11]
        gr_results.append(results)
        print 'Finished result for trace length = ',end
        
    return gr_results
    
def merge_traces(trace_list):
    merged_traces = {}
    for param in trace_list[0]:
        merged_traces[param] = []
        for chain in range(len(trace_list[0][param])):
            first_trace = trace_list[0][param][chain]
            concatenated_traces = np.concatenate((first_trace, trace_list[1][param][chain]))
            if len(trace_list) > 2:
                for other_trace in range(2, len(trace_list)):
                    concatenated_traces = np.concatenate((concatenated_traces, trace_list[other_trace][param][chain]))
            merged_traces[param].append(concatenated_traces)
    
    return merged_traces
    
def print_convergence_summary(param_trace):
    gr_results = gelman_rubin_trace_dict(param_trace)
    params1_2 = 0
    params1_1 = 0
    for param in gr_results.keys():
        if gr_results[param] < 1.2:
            params1_2 += 1
        if gr_results[param] < 1.1:
            params1_1 += 1
    
    perc_12 = (float(params1_2)/len(param_trace))*100
    perc_11 = (float(params1_1)/len(param_trace))*100
    
    print 'Number of parameters with GR below 1.2: ',params1_2,' of ',len(param_trace),' parameters.'
    print 'Percent of parameters with GR below 1.2: ',perc_12
    print 'Number of parameters with GR below 1.1: ',params1_1,' of ',len(param_trace),' parameters.'
    print 'Percent of parameters with GR below 1.1: ',perc_11
    
    return [params1_2, perc_12, params1_1, perc_11]
    
def convert_bayessb_output_to_tracedict(model, chain_prefix=None, prefix=None, accepts_suffix=None, tests_suffix=None, nchains=3, other_keys = None):
    if prefix == None:
        prefix = ''
    
    if chain_prefix == None:
        chain_prefix = ''
        
    if accepts_suffix == None:
        accepts_suffix = '_accepted_position_locations.npy'
    
    if tests_suffix == None:
        tests_suffix = '_alltestedpositions.npy'
    
    trace_dict = {}
    trace_dict['params'] = []
    new_keys = []
    if other_keys != None:
        for key in other_keys:
            key = re.sub('_', '', key)
            key = re.sub('.npy', '', key)
            trace_dict[key] = []
            new_keys.append(key)
    
    starting_point = []
    for param in model.parameters_rules():
        starting_point.append(np.log10(param.value))
    
    for chain in range(1, nchains+1):
        accepts_filename = chain_prefix+str(chain)+prefix+accepts_suffix
        tests_filename = chain_prefix+str(chain)+prefix+tests_suffix
        
        accepts = np.load(accepts_filename)
        tests = np.load(tests_filename)
        
        accepted_positions = np.zeros(np.shape(tests))
        
        last_accept = starting_point        
        
        for position in range(len(tests)):
            if accepts[position] == True:
                accepted_positions[position] = tests[position]
                last_accept = tests[position]
            
            else:
                accepted_positions[position] = last_accept
        
        trace_dict['params'].append(accepted_positions)
        last_accept = starting_point
        
        if other_keys != None:
            for old_key, new_key in zip(other_keys, new_keys):
                filename = chain_prefix+str(chain)+prefix+old_key
                
                data = np.load(filename)
                
                trace_dict[new_key].append(data) 
    
    return trace_dict
        