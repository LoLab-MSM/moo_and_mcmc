import pymc as pm
import numpy as np
import pysb.integrate
import scipy.interpolate
import os
import theano.tensor as t
import theano
import pickle
from pymc.backends import text

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
solver = pysb.integrate.Solver(earm, tspan, integrator='vode', rtol=1e-7, atol=1e-7, nsteps=10000)

@theano.compile.ops.as_op(itypes=[t.dvector],otypes=[t.dvector, t.dvector, t.dvector, t.dscalar, t.dscalar, t.dscalar]) #to use gpu use type t.fvector for all inputs/outputs
def likelihood(param_vector):
    # Sub in parameter values for current location in parameter space and simulate
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
    #error = e1_mBid + e1_cPARP + e2
    #print 'subbed values: ',[np.log10(param.value) for param in earm.parameters_rules()]
    #print 'mBid error: ',e1_mBid
    #print 'e1_cPARP: ',e1_cPARP
    #print 'e2: ',e2
    
    return sim_mBid, sim_cPARP, momp_sim, np.array(e1_mBid), np.array(e1_cPARP), np.array(e2) #to use gpu add .astype('float32') to end of first two arrays
    #return np.array(error)

#Setting up PyMC model
with model:
    #Add PySB rate parameters as unobserved random variables to PyMC model
    # To use gpu pass dtype='float32' and add .astype('float32') to end of np.log10(param.value)
    #params = pm.Normal('params', mu=[np.log10(param.value) for param in earm.parameters_rules()], sd=np.array([1.0]*len(earm.parameters_rules())), shape=(len(earm.parameters_rules())))
    lower_limits = np.zeros(len(earm.parameters_rules()))
    upper_limits = np.zeros(len(earm.parameters_rules()))
    starting_vals = np.zeros(len(earm.parameters_rules()))
    kf_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kf' in param.name]
    kr_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kr' in param.name]
    kc_idx = [idx for idx, param in enumerate(earm.parameters_rules()) if 'kc' in param.name]
    lower_limits[kf_idx] = -16    
    upper_limits[kf_idx] = -1
    #Sampling for kr is really KD values that are then used with the sampled kf to choose a kr
    lower_limits[kr_idx] = -3
    upper_limits[kr_idx] = 15
    lower_limits[kc_idx] = -6
    upper_limits[kc_idx] = 3
    starting_vals[kf_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kf' in param.name])
    #print 'starting kf vals: ',10**starting_vals[kf_idx]
    starting_vals[kr_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kr' in param.name])-starting_vals[kf_idx]
    #print 'starting kr vals: ',10**starting_vals[kr_idx]    
    starting_vals[kc_idx] = np.log10([param.value for param in earm.parameters_rules() if 'kc' in param.name])
    #print 'starting kc vals: ',10**starting_vals[kc_idx]    
    params = pm.Uniform('params', lower=lower_limits, upper=upper_limits, shape=(len(earm.parameters_rules())))    
    
    #Create dictionary of parameter locations in vector and names for use in likelihood function    
    name_dict = {i: param.name for i, param in enumerate([param for param in earm.parameters_rules()])}    
        
    #Add model likelihood - compare simulation to experimental data
    sim_icrp, sim_ecrp, sim_momp, icrp, ecrp, momp = likelihood(model.params)    
    #error = likelihood(model.params)
    
    pm.Normal('icrp_obs', mu=sim_icrp, sd=exp_data['nrm_var_ICRP'], observed=exp_data['norm_ICRP'])
    pm.Normal('ecrp_obs', mu=sim_ecrp, sd=exp_data['nrm_var_ECRP'], observed=exp_data['norm_ECRP'])
    pm.Normal('momp_obs', mu=sim_momp, sd=momp_var, observed=momp_data)    
    

    icrp = pm.Deterministic('icrp', icrp)
    ecrp = pm.Deterministic('ecrp', ecrp)
    momp = pm.Deterministic('momp', momp)
    #error_like = pm.ArbLikelihood('like', error)    
    
    #Select point in parameter space to start
    starting_pts = np.ones(len(starting_vals)) * np.random.randn(3, len(starting_vals)) + starting_vals
    for npt, pt in enumerate(starting_pts):
       while np.any(pt < lower_limits) or np.any(pt > upper_limits):
           starting_pts[npt] = np.ones(len(starting_vals)) * np.random.randn(1, len(starting_vals)) + starting_vals
    starts = [{'params':starting_pts[chain]} for chain in range(3)]
    starts[0]['params'] = starting_vals 
    #print 'starts: ',starts
    #Select stepping method
    nseedchains = 10*len(earm.parameters_rules())
    step = pm.Dream(variables=[model.params], nseedchains=nseedchains, verbose=True, blocked=True, start_random=False, save_history=True, parallel=False, model_name='earm_mtdreamzs_3chain_uni')
    
    #old_trace = text.load('2015_04_30_earm_direct_mtdreamzs_normal_prior')
    trace = pm.sample(30000, step, start=starts, njobs=3, use_mpi=False) #pass njobs=None to start multiple chains on different cpus
    
    text.dump('earm_mtdreamzs_3chain_uni', trace)    
    #text.dump('test', trace)       
    
    dictionary_to_pickle = {}

    for dictionary in trace:
        for var in dictionary:
            dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('earm_mtdreamzs_3chain_uni.p', 'wb'))
    #pickle.dump(dictionary_to_pickle, open('test.p', 'wb'))
    
#    from helper_fxns import convert_param_vec_dict_to_param_dict
#    from helper_fxns import merge_traces
#    from helper_fxns import print_convergence_summary
#    
#    old_traces = pickle.load(open('2015_04_30_earm_direct_mtdreamzs_normal_prior_merged_traces_95000.p'))
#    trace_list = [old_traces, dictionary_to_pickle]
#    merged_traces = merge_traces(trace_list)
#    
#    pickle.dump(merged_traces, open('2015_05_01_earm_direct_mtdreamzs_normal_prior_merged_traces_110000.p', 'wb'))
#    
#    trace_just_params = merged_traces
#    del trace_just_params['icrp_output']
#    del trace_just_params['ecrp_output']
#    del trace_just_params['momp_output']
#    del trace_just_params['icrp']
#    del trace_just_params['ecrp']
#    del trace_just_params['momp']
#    param_vec_dict = convert_param_vec_dict_to_param_dict(trace_just_params, earm.parameters_rules())
#    print_convergence_summary(param_vec_dict)
    
    
    
    
        

