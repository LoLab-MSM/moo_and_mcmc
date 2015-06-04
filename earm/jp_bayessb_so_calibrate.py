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
import pylab as plt

randomseed = 13
output_file_prefix = ''    
walk_length = 5000
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value,
              model.parameters['PARP_0'].value]

earm_path = '/home/pinojc/Copy/git/earm'
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

ntimes = len(exp_data['Time'])
tmul = 10
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes-1) * tmul + 1)

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
k_ids = [p.value for p in model.parameters_rules()]
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rate_mask])
bounds_radius = 2
solver = pysb.integrate.Solver(model, tspan, integrator='vode', rtol=1e-6, atol=1e-6,)


def likelihood(mcmc,position):
    Y=np.copy(position)
    param_values[rate_mask] = 10 ** Y
    changes={}
    changes['Bid_0'] = 0
    solver.run(param_values,initial_changes=changes)
    ysim_momp = solver.yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
        ysim_momp_norm = ysim_momp
    else:
        return 100000,
    solver.run(param_values)
    for obs_name, data_name, var_name, obs_total in \
            zip(obs_names, data_names, var_names, obs_totals):
        ysim = solver.yobs[obs_name][::tmul]
        ysim_norm = ysim / obs_total
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        if obs_name == 'mBid':
            e1 = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)
        else:
            e2 = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)
    ysim_momp = solver.yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
        print 'No aSmac!'
        ysim_momp_norm = ysim_momp
        t10 = 0
        t90 = 0
    else:
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
        try:
            t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0
    td = (t10 + t90) / 2
    ts = t90 - t10
    yfinal = ysim_momp[-1]
    momp_sim = [td, ts, yfinal]
    e3 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3
    error = e1 + e2 +e3
    try:
        totalerror[mcmc.iter] = error
        return error
    except AttributeError:
        return error

def prior(mcmc, position):
    """Distance to original parameter values"""
    return np.sum((position - prior_mean) ** 2 / ( 2 * prior_var))

def step(mcmc):
    """Print out some statistics every 20 steps"""
    if mcmc.iter % 100 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, float(mcmc.acceptance)/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)
    

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
#opts.anneal_length = walk_length

scenario = 1
opts.estimate_params = model.parameters_rules()

prior_mean = [np.log10(p.value) for p in opts.estimate_params]
prior_var =  2.0
totalerror = np.zeros(opts.nsteps)
error1 = np.zeros(opts.nsteps)
error2 = np.zeros(opts.nsteps)

mcmc = bayessb.MCMC(opts)

print 'Starting time:', strftime("%a, %d %b %Y %I:%M:%S")
mcmc.run()
print 'Ending time:', strftime("%a, %d %b %Y %I:%M:%S")
plt.semilogy(mcmc.likelihoods)
plt.show()
np.savetxt('mc_%s.txt' %randomseed,mcmc.likelihoods)
quit()
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

