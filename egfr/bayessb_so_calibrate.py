import pysb.integrate
import numpy as np
import pickle
import os
import bayessb_new as bayessb
from time import strftime
from optparse import OptionParser

from egfr.erbb_exec import model

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
#model.odes = pickle.load(open("pickled_inputs/erbb_simplified_odes.p", "rb"))
#model.species = pickle.load(open("pickled_inputs/erbb_simplified_species.p", "rb"))
#model.observables = pickle.load(open("pickled_inputs/erbb_simplified_observables.p", "rb"))
#model.reactions = pickle.load(open("pickled_inputs/erbb_simplified_reactions.p", "rb"))
#model.reactions_bidirectional = pickle.load(open("pickled_inputs/erbb_simplified_reactions_bidirectional.p", "rb"))

#Load in fitted parameter values
#parameter_file = 'fitted_parameters/earm_fitted_parameters_dictionary.p'
#with open(parameter_file, 'rb') as handle:
#    fittedparams = pickle.loads(handle.read())

#for i in range(len(model.parameters)):
#    if model.parameters[i].name in fittedparams:
#        model.parameters[i].value = fittedparams[model.parameters[i].name]

# Load experimental data file on ACCRE
egfr_path = '/home/shockle/egfr/egfr/experimental_data'
data_path = os.path.join(egfr_path,'experimental_data_A431_highEGF_unnorm.npy')
exp_data = np.load(data_path)
data_path = os.path.join(egfr_path, 'experimental_data_var_A431_highEGF_unnorm.npy')
exp_data_var = np.load(data_path)

tspan = np.linspace(0, 7200, num=36000)
t_exp_pts = np.array([0, 749, 1499, 2249, 2999, 4499, 8999, 13499, 17999, 35999])

obs_names = ['obsAKTPP', 'obsErbB1_P_CE', 'obsERKPP']

def extract_records(recarray, names):
    """Convert a record-type array and list of names into a float array"""
    return np.vstack([recarray[name] for name in names]).T

def likelihood(mcmc, position):
    """Distance between model trajectories and experimental data"""
    ysim = mcmc.simulate(position, observables=True)
    ysim_array = extract_records(ysim, obs_names)
    ysim_slice = ysim_array[t_exp_pts]
    try:
        objAKT[mcmc.iter] = np.sum((exp_data[:,0] - ysim_slice[:,0]) ** 2 / (2 * exp_data_var[:,0] ** 2))
        objErb[mcmc.iter] = np.sum((exp_data[:,1] - ysim_slice[:,1]) ** 2 / (2 * exp_data_var[:,1] ** 2))
        objERK[mcmc.iter] = np.sum((exp_data[:,2] - ysim_slice[:,2]) ** 2 / (2 * exp_data_var[:,2] ** 2))
        total_obj[mcmc.iter] = objAKT[mcmc.iter] + objErb[mcmc.iter] + objERK[mcmc.iter]
        return total_obj[mcmc.iter]
    except AttributeError:
         objective = np.sum((exp_data - ysim_slice) ** 2 / (2 * exp_data_var ** 2))
         return objective
     
def prior(mcmc, position):
    """Distance to original parameter values"""
    return np.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 1 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)
    #Save current data to file every 500 steps
    if mcmc.iter % 500 == 0:
        print 'Saving data to file'
        np.savetxt(output_file_prefix+'alltestedpositions.txt', mcmc.positions)
        np.savetxt(output_file_prefix+'totalobj.txt', total_obj)
        np.savetxt(output_file_prefix+'sobj_erk.txt', objERK)
        np.savetxt(output_file_prefix+'sobj_akt.txt', objAKT)
        np.savetxt(output_file_prefix+'sobj_erb.txt', objErb)
        np.savetxt(output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)

#Set BayesSB parameters
opts = bayessb.MCMCOpts()
opts.model = model
opts.tspan = tspan
opts.anneal_length = walk_length
opts.likelihood_fn = likelihood
opts.step_fn = step
opts.prior_fn = prior
opts.nsteps = walk_length
opts.seed = randomseed
opts.integrator = 'lsoda'
opts.atol=1e-7
opts.rtol=1e-7
opts.intsteps = 5000

#Initialize arrays for recording objective function values for each variable (AKT, ErbB1, and ERK) and the total objective
objAKT = np.zeros(opts.nsteps)
objErb = np.zeros(opts.nsteps)
objERK = np.zeros(opts.nsteps)
total_obj = np.zeros(opts.nsteps)

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
np.savetxt(output_file_prefix+'totalobj.txt', total_obj)
np.savetxt(output_file_prefix+'sobj_erk.txt', objERK)
np.savetxt(output_file_prefix+'sobj_akt.txt', objAKT)
np.savetxt(output_file_prefix+'sobj_erb.txt', objErb)
np.savetxt(output_file_prefix+'accepted_position_locations.txt', mcmc.accepts)

