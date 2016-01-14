# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:56:30 2016

@author: Erin
"""

from core import run_dream
from earm.lopez_embedded import model as earm
from earm_exp_parameters import earm_rates
from scipy.stats import norm
import os
from pysb.integrate import Solver
import numpy as np
from scipy import interpolate
from parameters import NormalParam, UniformParam
import pickle

# Info for loading/simulating EARM PySB model and experimental data
# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [earm.parameters['Bid_0'].value,
              earm.parameters['PARP_0'].value]

# Load experimental data file
earm_path = '/home/shockle/earm'
#earm_path = '/Users/Erin/git/earm'
data_path = os.path.join(earm_path, 'xpdata', 'forfits',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_obs_total = earm.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

# Build time points for the integrator, using the same time scale as the
# experimental data but with greater resolution to help the integrator converge.
ntimes = len(exp_data['Time'])
# Factor by which to increase time resolution
tmul = 100
# Do the sampling such that the original experimental timepoints can be
# extracted with a slice expression instead of requiring interpolation.
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes-1) * tmul + 1)
# Initialize solver object
solver = Solver(earm, tspan, integrator='vode', rtol=1e-7, atol=1e-7, nsteps=10000)

def likelihood(param_vector):
    for i in range(len(param_vector)):
       if i in kr_idx:
           #Sampled value is a KD value that is then used with the kf to choose a kr
           earm.parameters_rules()[name_dict[i]].value = 10**(param_vector[i]+param_vector[i-1])
        #    earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]
            #print 'KD value = ',10**param_vector[i]
            #print 'set parameter: ',earm.parameters_rules()[name_dict[i]].name,' to ',10**(param_vector[i]+param_vector[i-1])
    else:
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]
            #print 'set parameter: ',earm.parameters_rules()[name_dict[i]].name,' to ',10**param_vector[i]
        #earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]
    #print 'subbed kf vals: ',10**param_vector[kf_idx]
    #print 'subbed kr vals: ',10**param_vector[kr_idx]
    #print 'subbed kc vals: ',10**param_vector[kc_idx]
    solver.run()
    
    e1 = {}
    sims = {}
    for obs_name, data_name, var_name, obs_total in \
            zip(obs_names, data_names, var_names, obs_totals):
        # Get model observable trajectory (this is the slice expression
        # mentioned above in the comment for tspan)
        ysim = solver.yobs[obs_name][::tmul]
        # Normalize it to 0-1
        ysim_norm = ysim / obs_total
        # Get experimental measurement and variance
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        # Compute error between simulation and experiment (chi-squared)
        e1[obs_name] = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)    
        sims[obs_name] = ysim_norm
    
    e1_mBid = e1['mBid'] 
    e1_mBid = -np.log10(e1_mBid)
    if np.isnan(e1_mBid):
        e1_mBid = -np.inf
    e1_cPARP = e1['cPARP']
    e1_cPARP = -np.log10(e1_cPARP)
    if np.isnan(e1_cPARP):
        e1_cPARP = -np.inf
    sim_mBid = sims['mBid']
    if np.any(np.isnan(sim_mBid)):
        sim_mBid.fill(-np.inf)
    sim_cPARP = sims['cPARP']
    if np.any(np.isnan(sim_cPARP)):
        sim_cPARP.fill(-np.inf)
    # Calculate Td, Ts, and final value for IMS-RP reporter
    # =====
    # Normalize trajectory
    ysim_momp = solver.yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
        ysim_momp_norm = ysim_momp
        t10 = 0
        t90 = 0
    
    else:  
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        # Build a spline to interpolate it
        st, sc, sk = interpolate.splrep(solver.tspan, ysim_momp_norm)
        try: 
            # Use root-finding to find the point where trajectory reaches 10% and 90%
            t10 = interpolate.sproot((st, sc-0.10, sk))[0]
            t90 = interpolate.sproot((st, sc-0.90, sk))[0]
        #If integration has failed and nans are present in trajectory, 
        # interpolation will fail and an IndexError will occur
        except IndexError:
            t10 = 0
            t90 = 0
    # Calculate Td as the mean of these times
    td = (t10 + t90) / 2
    # Calculate Ts as their difference
    ts = t90 - t10
    # Get yfinal, the last element from the trajectory
    yfinal = ysim_momp[-1]
    # Build a vector of the 3 variables to fit
    momp_sim = np.array([td, ts, yfinal]) #to use gpu add type='float32'
    
    # Perform chi-squared calculation against mean and variance vectors
    e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3
    e2 = -np.log10(e2)
    if np.isnan(e2):
        e2 = -np.inf
        momp_sim.fill(-np.inf)
    #error = e1_mBid + e1_cPARP + e2
    #print 'subbed values: ',[np.log10(param.value) for param in earm.parameters_rules()]
    #print 'mBid error: ',e1_mBid
    #print 'e1_cPARP: ',e1_cPARP
    #print 'e2: ',e2
    #print 'sim mBid shape: ',sim_mBid.shape
    #print 'sim cPARP shape: ',sim_cPARP.shape
    #print 'sim MOMP shape: ',momp_sim.shape
    logp = np.sum(norm.logpdf(sim_mBid, exp_data['norm_ICRP'], exp_data['nrm_var_ICRP'])) + np.sum(norm.logpdf(sim_cPARP, exp_data['norm_ECRP'], exp_data['nrm_var_ECRP'])) + np.sum(norm.logpdf(momp_sim, momp_data, momp_var))
        
    return logp

uni_param_dict = {}    
norm_param_dict = {}
uni_n = 0
norm_n = 0
uni_idx = []
norm_idx = []
kr_idx = []
name_dict = {}
for i, param in enumerate(earm.parameters_rules()):
    name_dict[i] = param.name
    if earm_rates[param.name]['type'] == 'uniform':
        uni_idx.append(i)
        uni_param_dict[param.name] = (i, uni_n)
        uni_n += 1
    elif earm_rates[param.name]['type'] == 'normal':
        norm_idx.append(i)
        norm_param_dict[param.name] = (i, norm_n)
        norm_n += 1
    if 'kr' in param.name:
        kr_idx.append(i)
uni_idx = np.array(uni_idx)
norm_idx = np.array(norm_idx)
n_uni_params = uni_n
n_norm_params = norm_n
lower_limits = np.zeros(n_uni_params)
upper_limits = np.zeros(n_uni_params)
means = np.zeros(n_norm_params)
sds = np.zeros(n_norm_params)
starting_vals = np.array([param.value for param in earm.parameters_rules()])
for key in uni_param_dict:
    lower_limits[uni_param_dict[key][1]] = earm_rates[key]['lower']
    upper_limits[uni_param_dict[key][1]] = earm_rates[key]['upper']
for key in norm_param_dict:
    means[norm_param_dict[key][1]] = np.log10(earm_rates[key]['mean'])
    sds[norm_param_dict[key][1]] = earm_rates[key]['sd'] 

uniform_params = UniformParam('uniform_params', starting_vals[uni_idx], lower_limits, upper_limits)
normal_params = NormalParam('normal_params', starting_vals[norm_idx], means, sds)

sampled_params, log_ps = run_dream([uniform_params, normal_params], likelihood, niterations=10, snooker=0)

for chain in range(len(sampled_params)):
    np.save('earm_dreamz_3chain_a_sampled_params_chain_'+str(chain), sampled_params[chain])
    np.save('earm_dreamz_3chain_a_logps_chain_'+str(chain), log_ps[chain])

