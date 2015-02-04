import pymc as pm
import numpy as np
import pysb.integrate
import scipy.interpolate
import os
from theano.printing import Print
import theano.tensor as t
from theano import function
from theano import shared
import theano
import pickle

from earm.lopez_embedded import model as earm

model = pm.Model()

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
earm_path = '/Users/Erin/git/earm'
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
solver = pysb.integrate.Solver(earm, tspan, rtol=1e-7, atol=1e-7)

@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dvector, t.dvector, t.dvector]) #to use gpu use type t.fvector for all inputs/outputs
def likelihood(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]
    print 'running the solver'
    solver.run()
    
    # Get model observable trajectory (this is the slice expression
    # mentioned above in the comment for tspan)
    ysim = solver.yobs['mBid'][::tmul]
    # Normalize it to 0-1
    ysim_norm_bid = ysim / obs_totals[0]
    
    # Get model observable trajectory (this is the slice expression
    # mentioned above in the comment for tspan)
    ysim = solver.yobs['cPARP'][::tmul]
    # Normalize it to 0-1
    ysim_norm_parp = ysim / obs_totals[1]
    
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
    
    return ysim_norm_bid, ysim_norm_parp, momp_sim #to use gpu add .astype('float32') to end of first two arrays
    
@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dvector])
def likelihood_icrp(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

    solver.run()
    
    # Get model observable trajectory (this is the slice expression
    # mentioned above in the comment for tspan)
    ysim = solver.yobs['mBid'][::tmul]
    # Normalize it to 0-1
    ysim_norm = ysim / obs_totals[0]
    print ysim_norm
    
    return ysim_norm

@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dvector])
def likelihood_ecrp(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

    solver.run()
    
    # Get model observable trajectory (this is the slice expression
    # mentioned above in the comment for tspan)
    ysim = solver.yobs['cPARP'][::tmul]
    # Normalize it to 0-1
    ysim_norm = ysim / obs_totals[1]
    print ysim_norm
    
    return ysim_norm

@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dvector])
def likelihood_momp(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

    solver.run()
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
    momp_sim = np.array([td, ts, yfinal])
    return momp_sim


#Setting up PyMC model
with model:
    #Add PySB rate parameters as unobserved random variables to PyMC model
    # To use gpu pass dtype='float32' and add .astype('float32') to end of np.log10(param.value)
    len_non_kf_params = len([param for param in earm.parameters_rules() if 'kf' not in param.name])
    pm.Normal('params', mu=[np.log10(param.value) for param in earm.parameters_rules() if 'kf' not in param.name], sd=np.array([.2]*len_non_kf_params), shape=(len_non_kf_params))
    #for param in robertson.parameters_rules():
    #    pm.Normal(param.name, mu=np.log10(param.value), sd=.2)
    
    #Create dictionary of parameter locations in vector and names for use in likelihood function    
    name_dict = {i: param.name for i, param in enumerate([param for param in earm.parameters_rules() if 'kf' not in param.name])}    
        
    #Add model likelihood - compare simulation to experimental data
    icrp, ecrp, momp = likelihood(model.params)    
    
    pm.Normal('icrp_output', mu=icrp, sd=np.sqrt(exp_data['nrm_var_ICRP']), observed=exp_data['norm_ICRP'])
    pm.Normal('ecrp_output', mu=ecrp, sd=np.sqrt(exp_data['nrm_var_ECRP']), observed=exp_data['norm_ECRP'])    
    pm.Normal('momp_output', mu=momp, sd=np.sqrt(momp_var), observed=momp_data)
    
    #Select point in parameter space to start
    #start = pm.find_MAP()
    
    #Select stepping method
    nseedchains = 10*len_non_kf_params
    step = pm.Dream(nseedchains=nseedchains, blocked=True, snooker=0)
    
    trace = pm.sample(425000, step, njobs=5) #pass njobs=None to start multiple chains on different cpus
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('2015_02_03_earm_test.p', 'wb'))
    
        

