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
from pymc.backends import text
from mpi4py import MPI

from earm.lopez_embedded import model as earm

model = pm.Model()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
solver = pysb.integrate.Solver(earm, tspan, integrator='vode', rtol=1e-7, atol=1e-7, nsteps=10000)

cov = np.identity(len(earm.parameters_rules()))
mu = np.array([np.log10(param.value) for param in earm.parameters_rules()])
m = np.random.multivariate_normal(mu, cov, size=10*len(earm.parameters_rules()))
np.save('earm_embedded_seed.npy', m)

@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dscalar, t.dscalar, t.dscalar]) #to use gpu use type t.fvector for all inputs/outputs
def likelihood(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
    for i in range(len(param_vector)):
        earm.parameters_rules()[name_dict[i]].value = 10**param_vector[i]

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
    
    return np.array(e1_mBid), np.array(e1_cPARP), np.array(e2) #to use gpu add .astype('float32') to end of first two arrays
    #return np.array(error)

#Setting up PyMC model
with model:
    #Add PySB rate parameters as unobserved random variables to PyMC model
    # To use gpu pass dtype='float32' and add .astype('float32') to end of np.log10(param.value)
    len_non_kf_params = len([param for param in earm.parameters_rules() if 'kf' not in param.name])
    params = pm.Normal('params', mu=[np.log10(param.value) for param in earm.parameters_rules()], sd=np.array([1.0]*len(earm.parameters_rules())), shape=(len(earm.parameters_rules())))
    #pm.Uniform('params', lower=np.array([-5.0]*len(earm.parameters_rules())), upper=np.array([8.0]*len(earm.parameters_rules())), shape=(len(earm.parameters_rules())))    
    #for param in robertson.parameters_rules():
    #    pm.Normal(param.name, mu=np.log10(param.value), sd=.2)
    
    #Create dictionary of parameter locations in vector and names for use in likelihood function    
    name_dict = {i: param.name for i, param in enumerate([param for param in earm.parameters_rules()])}    
        
    #Add model likelihood - compare simulation to experimental data
    icrp, ecrp, momp = likelihood(model.params)    
    #error = likelihood(model.params)
    
    icrp_like = pm.ArbLikelihood('icrp_output', icrp)
    ecrp_like = pm.ArbLikelihood('ecrp_output', ecrp)    
    momp_like = pm.ArbLikelihood('momp_output', momp)
    
    icrp = pm.Deterministic('icrp', icrp)
    ecrp = pm.Deterministic('ecrp', ecrp)
    momp = pm.Deterministic('momp', momp)
    #error_like = pm.ArbLikelihood('like', error)    
    
    #Select point in parameter space to start
    #start = pm.find_MAP()
    
    #Select stepping method
    nseedchains = 10*len(earm.parameters_rules())
    step = pm.Dream(variables=[model.params], nseedchains=nseedchains, blocked=True, multitry=5, start_random=False, save_history=True, parallel=False, adapt_crossover=False, history_file='2015_04_28_earm_embedded_mtdreamzs_normal_prior_history.npy', crossover_file='2015_04_18_earm_embedded_mtdreamzs_normal_prior_crossovervals.npy')

    old_trace = text.load('2015_04_28_earm_embedded_mtdreamzs_normal_prior')
    trace = pm.sample(15000, step, njobs=3, trace=old_trace, use_mpi=False) #pass njobs=None to start multiple chains on different cpus
    
    text.dump('2015_04_29_earm_embedded_mtdreamzs_normal_prior', trace)    
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
            dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('2015_04_29_earm_embedded_mtdreamzs_normal_prior.p', 'wb'))
    
    from helper_fxns import convert_param_vec_dict_to_param_dict
    from helper_fxns import merge_traces
    from helper_fxns import print_convergence_summary
    
    old_traces = pickle.load(open('2015_04_28_earm_embedded_mtdreamzs_normal_prior_merged_traces_50000.p'))
    trace_list = [old_traces, dictionary_to_pickle]
    merged_traces = merge_traces(trace_list)
    
    pickle.dump(merged_traces, open('2015_04_29_earm_direct_mtdreamzs_normal_prior_merged_traces_65000.p', 'wb'))
    
    trace_just_params = merged_traces
    del trace_just_params['icrp_output']
    del trace_just_params['ecrp_output']
    del trace_just_params['momp_output']
    del trace_just_params['icrp']
    del trace_just_params['ecrp']
    del trace_just_params['momp']
    param_vec_dict = convert_param_vec_dict_to_param_dict(trace_just_params, earm.parameters_rules())
    print_convergence_summary(param_vec_dict)
                        
    
    
        

