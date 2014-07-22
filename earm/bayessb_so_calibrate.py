import pysb.integrate
import numpy as np
import scipy.interpolate
import pickle
import os
import bayessb_new as bayessb
from time import strftime
import socket
from optparse import OptionParser

from earm.lopez_embedded import model

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

#Load ODES and Jacobian from pickled values
#model.odes = pickle.load(open("pickled_inputs/earm_lopez_embedded_odes.p", "rb"))
#model.jacobian = pickle.load(open("pickled_inputs/earm_lopez_embedded_jacobian.p", "rb"))
#model.species = pickle.load(open("pickled_inputs/earm_lopez_embedded_species.p", "rb"))
#model.observables = pickle.load(open("pickled_inputs/earm_lopez_embedded_observables.p", "rb"))
#model.reactions = pickle.load(open("pickled_inputs/earm_lopez_embedded_reactions.p", "rb"))
#model.reactions_bidirectional = pickle.load(open("pickled_inputs/earm_lopez_embedded_reactions_bidirectional.p", "rb"))

#Load in fitted parameter values
parameter_file = 'fitted_parameters/earm_fitted_parameters_dictionary.p'
with open(parameter_file, 'rb') as handle:
    fittedparams = pickle.loads(handle.read())

for i in range(len(model.parameters)):
    if model.parameters[i].name in fittedparams:
        model.parameters[i].value = fittedparams[model.parameters[i].name]

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
if socket.gethostname() == 'localhost':
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
tmul = 500
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

#Likelihood function is objective function from earm.estimate_m1a.py with small changes
def likelihood(mcmc, position):

    # Simulate model at current position in parameter space
    yobs = mcmc.simulate(position, observables=True)
    # Calculate error for point-by-point trajectory comparisons
    e1 = 0
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
        e1 += np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)

    # Calculate error for Td, Ts, and final value for IMS-RP reporter
    # =====
    # Normalize trajectory
    ysim_momp = yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
	print 'No aSmac!'
	ysim_momp_norm = ysim_momp
	t10 = 0
	t90 = 0
    else:
    	ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
   	# Build a spline to interpolate it
   	st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
   	# Use root-finding to find the point where trajectory reaches 10% and 90%
   	try:
		t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
   		t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
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
    momp_sim = [td, ts, yfinal]
    # Perform chi-squared calculation against mean and variance vectors
    e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3

    error = e1 + e2
    try:
        totalerror[mcmc.iter] = error
        error1[mcmc.iter] = e1
        error2[mcmc.iter] = e2
        return error
    except AttributeError:
        return error

def prior(mcmc, position):
    """Distance to original parameter values"""
    return np.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 1 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)
    if mcmc.iter % 1000 == 0:
        np.savetxt(str(iter)+output_file_prefix+'alltestedpositions.txt', mcmc.positions)
        np.savetxt(str(iter)+output_file_prefix+'totalobj.txt', totalerror)
        np.savetxt(str(iter)+output_file_prefix+'sobj_error1.txt', error1)
        np.savetxt(str(iter)+output_file_prefix+'sobj_error2.txt', error2)
        np.savetxt(str(iter)+output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)

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
opts.integrator = 'lsoda'
opts.atol=1e-7
opts.rtol=1e-7
opts.with_jacobian = True
opts.intsteps = 5000
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
prior_var =  3.0

#Initialize arrays to hold calculated objective function values
totalerror = np.zeros(opts.nsteps)
error1 = np.zeros(opts.nsteps)
error2 = np.zeros(opts.nsteps)

mcmc = bayessb.MCMC(opts)

print 'Starting time:', strftime("%a, %d %b %Y %I:%M:%S")
mcmc.run()
print 'Ending time:', strftime("%a, %d %b %Y %I:%M:%S")

#Save fitted parameter values and objective function values from calibration
fitted_values = mcmc.cur_params()[mcmc.estimate_idx]
param_dict = {}
for param, new_value in zip(opts.estimate_params, fitted_values):
    change = np.log10(new_value / param.value)
    values = (param.name, param.value, new_value, change)
    print '%-10s %-12.2g %-12.2g %-+6.2f' % values
    param_dict[param.name] = new_value

with open(output_file_prefix+'fittedparamdict.p', 'wb') as handle:
    pickle.dump(param_dict, handle)

name = [p.name for p in opts.estimate_params]
oldvalues = [p.value for p in opts.estimate_params]
newvalues = mcmc.cur_params()[mcmc.estimate_idx]
oldvalues_array = np.array(oldvalues)
newvalues_array = np.array(newvalues)
change = np.log10(newvalues_array / oldvalues_array)
combined = np.column_stack((name, oldvalues, newvalues, change))
np.savetxt(output_file_prefix+'fittedparams.txt', combined, delimiter=' ', fmt='%s')
np.savetxt(output_file_prefix+'alltestedpositions.txt', mcmc.positions)
np.savetxt(output_file_prefix+'totalobj.txt', totalerror)
np.savetxt(output_file_prefix+'sobj_error1.txt', error1)
np.savetxt(output_file_prefix+'sobj_error2.txt', error2)
np.savetxt(output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)

