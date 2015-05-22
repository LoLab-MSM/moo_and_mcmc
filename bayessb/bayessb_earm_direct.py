import pysb.integrate
import numpy as np
import scipy.interpolate
import pickle
import os
import bayessb_new as bayessb
from time import strftime
import socket
from optparse import OptionParser

from earm.lopez_direct import model

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

# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value,
              model.parameters['PARP_0'].value]

# Load experimental data file
if socket.gethostname() == 'Erins-MacBook-Pro.local':
    earm_path = '/Users/Erin/git/earm'
if socket.gethostname() == 'puma':
    # This is the puma host name
    earm_path = '/home/shockle/earm'
data_path = os.path.join(earm_path, 'xpdata', 'forfits',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
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
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes-1) * tmul + 1)

# Get parameters for rates only
rate_params = model.parameters_rules()
# Build a boolean mask for those params against the entire param list
rate_mask = np.array([p in rate_params for p in model.parameters])
# Build vector of nominal parameter values from the model
nominal_values = np.array([p.value for p in model.parameters])

def likelihood(mcmc, position):    

    yobs = mcmc.simulate(position, observables=True)    
    e1 = {}
    for obs_name, data_name, var_name, obs_total in \
            zip(obs_names, data_names, var_names, obs_totals):
        # Get model observable trajectory (this is the slice expression
        # mentioned above in the comment for tspan)
        ysim = yobs[obs_name][::tmul]
        # Normalize it to 0-1
        ysim_norm = ysim / obs_total
        # Get experimental measurement and variance
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        # Compute error between simulation and experiment (chi-squared)
        e1[obs_name] = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)    
    
    e1_mBid = e1['mBid'] 
    e1_mBid = np.log(e1_mBid)
    if np.isnan(e1_mBid):
        e1_mBid = np.inf
    e1_cPARP = e1['cPARP']
    e1_cPARP = np.log(e1_cPARP)
    if np.isnan(e1_cPARP):
        e1_cPARP = np.inf

    # Calculate Td, Ts, and final value for IMS-RP reporter
    # =====
    # Normalize trajectory
    ysim_momp = yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
        ysim_momp_norm = ysim_momp
        t10 = 0
        t90 = 0
    
    else:  
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        # Build a spline to interpolate it
        st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
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
    e2 = np.log(e2)
    if np.isnan(e2):
        e2 = np.inf
        
    error = e1_mBid + e1_cPARP + e2

    try:
        totalerror[mcmc.iter] = error
        error1[mcmc.iter] = e1_mBid
        error2[mcmc.iter] = e1_cPARP
        error3[mcmc.iter] = e2
        return error
    except AttributeError:
        return error    

def prior(mcmc, position):
    """Normal probability density.  Chosen to be the same as PyMC's normal prior"""

    prob_dens = np.sum((-tau * (position - prior_mean) ** 2 + np.log(tau / np.pi / 2.)) / 2.)    
    
    return -prob_dens

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 1 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)
    if mcmc.iter % 100000 == 0:
        np.save(str(mcmc.iter)+output_file_prefix+'alltestedpositions.npy', mcmc.positions)
        np.save(str(mcmc.iter)+output_file_prefix+'totalobj.npy', totalerror)
        np.save(str(mcmc.iter)+output_file_prefix+'sobj_error1.npy', error1)
        np.save(str(mcmc.iter)+output_file_prefix+'sobj_error2.npy', error2)
        np.save(str(mcmc.iter)+output_file_prefix+'sobj_error3.npy', error3)
        np.save(str(mcmc.iter)+output_file_prefix+'accepted_position_locations.npy', mcmc.accepts)

#Set BayesSB parameters
opts = bayessb.MCMCOpts()
opts.model = model
opts.tspan = tspan
opts.anneal_length = 0
opts.likelihood_fn = likelihood
opts.step_fn = step
opts.prior_fn = prior
opts.nsteps = walk_length
opts.seed = randomseed
opts.integrator = 'vode'
opts.atol=1e-7
opts.rtol=1e-7
opts.intsteps = 10000
opts.T_init = 1

scenario = 1

# A few estimation scenarios:
if scenario == 1:
    # estimate rates only (not initial conditions)
    opts.estimate_params = model.parameters_rules()
elif scenario == 2:
    # use hessian
    opts.estimate_params = model.parameters_rules()
    # Warning: hessian-guidance is expensive when fitting many parameters -- the
    # time to calculate the hessian increases with the square of the number of
    # parameters to fit!
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 6
else:
    raise RuntimeError("unknown scenario number")

# values for prior calculation
prior_mean = [np.log10(p.value) for p in opts.estimate_params]
# prior_var is set to 3.0
sd =  1.0
tau = float(1/(sd**2))

#Initialize arrays to hold calculated objective function values
totalerror = np.zeros(opts.nsteps)
error1 = np.zeros(opts.nsteps)
error2 = np.zeros(opts.nsteps)
error3 = np.zeros(opts.nsteps)

mcmc = bayessb.MCMC(opts)

print 'Starting time:', strftime("%a, %d %b %Y %I:%M:%S")
mcmc.run()
print 'Ending time:', strftime("%a, %d %b %Y %I:%M:%S")

np.save(output_file_prefix+'alltestedpositions.npy', mcmc.positions)
np.save(output_file_prefix+'totalobj.npy', totalerror)
np.save(output_file_prefix+'sobj_error1.npy', error1)
np.save(output_file_prefix+'sobj_error2.npy', error2)
np.save(output_file_prefix+'sobj_error3.npy', error3)
np.save(output_file_prefix+'accepted_position_locations.npy', mcmc.accepts)
