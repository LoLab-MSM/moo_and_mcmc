# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:37:15 2015

@author: Erin
"""

from earm.lopez_embedded import model as earm
import emcee
import numpy as np
import os
import pysb.integrate
import scipy.interpolate

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
solver = pysb.integrate.Solver(earm, tspan, integrator='vode', rtol=1e-7, atol=1e-7, nsteps=10000)

kf_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kf' in param.name]
kr_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kr' in param.name]
kc_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kc' in param.name]
lower_limits = np.zeros(len(earm.parameters_rules()))
upper_limits = np.zeros(len(earm.parameters_rules()))
starting_vals = np.zeros(len(earm.parameters_rules()))
lower_limits[kf_idx] = -16    
upper_limits[kf_idx] = -1
#Sampling for kr is really KD values that are then used with the sampled kf to choose a kr
lower_limits[kr_idx] = -3
upper_limits[kr_idx] = 15
lower_limits[kc_idx] = -6
upper_limits[kc_idx] = 3
starting_vals[kf_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kf' in param.name])
starting_vals[kr_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kr' in param.name])-starting_vals[kf_idx]  
starting_vals[kc_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kc' in param.name])

name_dict = name_dict = {i: param.name for i, param in enumerate([param for param in earm.parameters_rules()])} 

icrp_tau = 1/(exp_data['nrm_var_ICRP']**2)
ecrp_tau = 1/(exp_data['nrm_var_ECRP']**2)
momp_tau = 1/(momp_var**2)

icrp_constant = np.log(icrp_tau/np.pi/2)
ecrp_constant = np.log(ecrp_tau/np.pi/2)
momp_constant = np.log(momp_tau/np.pi/2)

def log_prior(theta):
   if np.any(theta < lower_limits) or np.any(theta > upper_limits):
       return -np.inf
   
   else:
       return np.sum(-np.log(upper_limits-lower_limits))

def log_like(theta):
    param_vector = theta
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        if i in kr_idx:
            #Sampled value is a KD value that is then used with the kf to choose a kr
            earm.parameters_rules()[name_dict[i]].value = 10**(param_vector[i]+param_vector[i-1])

        else:
            earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

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
    sim_mBid = sims['mBid']
    if np.isnan(e1_mBid):
        e1_mBid = -np.inf
        sim_mBid.fill(-np.inf)      
    e1_cPARP = e1['cPARP']
    e1_cPARP = -np.log10(e1_cPARP)
    sim_cPARP = sims['cPARP']
    if np.isnan(e1_cPARP):
        e1_cPARP = -np.inf
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
        st, sc, sk = scipy.interpolate.splrep(solver.tspan, ysim_momp_norm)
        try: 
            # Use root-finding to find the point where trajectory reaches 10% and 90%
            t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
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
    
    icrp_logp = np.sum((-icrp_tau * (sim_mBid-exp_data['norm_ICRP']) **2 + icrp_constant)/2)
    ecrp_logp = np.sum((-ecrp_tau * (sim_cPARP-exp_data['norm_ECRP']) **2 + ecrp_constant)/2)
    momp_logp = np.sum((-momp_tau * (momp_sim - momp_data) **2 + momp_constant)/2)
    
    return icrp_logp + ecrp_logp + momp_logp
    
def log_post(theta):
    prior = log_prior(theta)
    like = log_like(theta)
    post = prior + like
    print 'prior: ',prior,' like: ',like,' post: ',post

    return post
    
ndim = len(earm.parameters_rules())
nwalkers = 250

start = [starting_vals + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, threads=16)

sampler.run_mcmc(start, 500)

np.save('earm_uniform_emcee_250_500.npy', sampler.chain)


    
    

       