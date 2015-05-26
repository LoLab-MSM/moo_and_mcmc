# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:44:29 2015

@author: Erin
"""

from earm.lopez_direct import model
import pickle
from pysb.integrate import Solver
import numpy as np
import os
import scipy

# Info for loading/simulating EARM PySB model and experimental data
# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value, model.parameters['PARP_0'].value]

# Load experimental data file
earm_path = '/home/shockle/earm'
data_path = os.path.join(earm_path, 'xpdata', 'forfits', 'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_obs_total = model.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

# Build time points for the integrator, using the same time scale as the
# experimental data but with greater resolution to help the integrator converge.
ntimes = len(exp_data['Time'])
# Factor by which to increase time resolution
tmul = 100
# Do the sampling such that the original experimental timepoints can be
# extracted with a slice expression instead of requiring interpolation.
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1], (ntimes-1) * tmul + 1)

solver = Solver(model, tspan, integrator='vode', rtol=1e-7, atol=1e-7, nsteps=10000)

name_dict = {i: param.name for i, param in enumerate([param for param in model.parameters_rules()])}    

def likelihood(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        model.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

    solver.run()
    
    e1 = {}
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
    
    e1_mBid = e1['mBid'] 
    e1_mBid = -np.log(e1_mBid)
    if np.isnan(e1_mBid):
        e1_mBid = -np.inf
    e1_cPARP = e1['cPARP']
    e1_cPARP = -np.log(e1_cPARP)
    if np.isnan(e1_cPARP):
        e1_cPARP = -np.inf
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
    e2 = -np.log(e2)
    if np.isnan(e2):
        e2 = -np.inf
    #error = e1_mBid + e1_cPARP + e2
    
    return np.array(e1_mBid), np.array(e1_cPARP), np.array(e2) 

mpi_output = pickle.load(open('testing_mpi.p'))

mpi_parameters = np.concatenate((mpi_output['params'][0], mpi_output['params'][1], mpi_output['params'][2]))
mpi_momp = np.concatenate((mpi_output['momp'][0], mpi_output['momp'][1], mpi_output['momp'][2]))
mpi_icrp = np.concatenate((mpi_output['icrp'][0], mpi_output['icrp'][1], mpi_output['icrp'][2]))
mpi_ecrp = np.concatenate((mpi_output['ecrp'][0], mpi_output['ecrp'][1], mpi_output['ecrp'][2]))

momp_output = np.zeros((np.shape(mpi_momp)))
icrp_output = np.zeros((np.shape(mpi_icrp)))
ecrp_output = np.zeros((np.shape(mpi_ecrp)))
momp_mismatches = np.zeros((np.shape(mpi_momp)))
icrp_mismatches = np.zeros((np.shape(mpi_icrp)))
ecrp_mismatches = np.zeros((np.shape(mpi_ecrp)))

for i, param_vec in enumerate(mpi_parameters):
    check_icrp, check_ecrp, check_momp = likelihood(param_vec)
    momp_output[i] = check_momp
    icrp_output[i] = check_icrp
    ecrp_output[i] = check_ecrp
    
    if momp_output[i] != mpi_momp[i]:
        print 'MOMP output does not match for parameter set '+str(i)+'.'
        momp_mismatches[i] = 1
    
    if ecrp_output[i] != mpi_ecrp[i]:
        print 'ECRP output does not match for parameter set '+str(i)+'.'
        ecrp_mismatches[i] = 1
    
    if icrp_output[i] != mpi_icrp[i]:
        print 'ICRP output does not match for parameter set '+str(i)+'.'
        icrp_mismatches[i] = 1


