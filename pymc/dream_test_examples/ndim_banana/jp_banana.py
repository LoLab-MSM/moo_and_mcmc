# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# A PyMC implementation of example 1 from MT-DREAM(ZS) original Matlab code.
# n-dimensional banana-shaped Gaussian distribution (example N=2)
import pylab as plt
import pymc as pm
import numpy as np
from pymc.backends import text
import pickle
import theano
import theano.tensor as t
from sklearn.cluster import KMeans
from _line_profiler import label
from scipy.cluster.vq import kmeans,vq
b = .1
log_F = -4.1405

C = np.identity(2)
C[0][0] = 100
invC = 1/C
invC[np.where(np.isinf(invC))] = 0

#Create initial samples matrix m that will be loaded in as DREAM history file
mu = np.zeros((2))
cov = 10*np.identity(2)
m = np.random.multivariate_normal(mean=mu, cov=cov, size=20)
np.save('ndim_banana_seed.npy', m)



@theano.compile.ops.as_op(itypes=[t.dvector], otypes=[t.dscalar])
def likelihood(params):
    new_param2 = params[1] + b * params[0]**2 - 100*b
    new_params = np.array([params[0], new_param2])
    log_L = log_F - .5 * np.sum(new_params*np.dot(invC, new_params))
    return log_L

def logp(params):
    params = model.params
    return log_F - .5 * np.sum(np.array([params[0], params[1] + b * params[0]**2 - 100*b])*np.dot(invC, np.array([params[0], params[1] + b * params[0]**2 - 100*b])))


def logp1(params):
    return -1*(log_F - .5 * np.sum(np.array([params[0], params[1] + b * params[0]**2 - 100*b])*np.dot(invC, np.array([params[0], params[1] + b * params[0]**2 - 100*b])))),
    #print log_L.eval()
    #print 'log_L: ',log_L  
    #return log_L

from refactored_pso import PSO
import time


nchains = 5

pso = PSO()
pso.update_w = False
pso.set_cost_function(logp1)
pso.set_start_position(np.array([-20,0]))
pso.set_bounds(range=100,lower=[-50,-100],upper=[50,50])
pso.set_speed(-100, 100)
pso.run(500,100)
ranked = pso.return_ranked_populations()
data = ranked
# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,nchains)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
np.save('ndim_banana_seed.npy',ranked)
# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or',
     data[idx==2,0],data[idx==2,1],'og',
     data[idx==3,0],data[idx==3,1],'oc',
     data[idx==4,0],data[idx==4,1],'ok')
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.xlim(-30,30)
plt.ylim(-60,15)
#plt.savefig('kmeans.png')
plt.show()


#quit()
# colors = np.zeros((len(pso.population),1))
# for n,part in enumerate(pso.population):
#     colors[n] = part.best.fitness.values[0]
# cm = plt.cm.get_cmap('RdYlBu')
# cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
# for part in pso.population:
#     plt.scatter(part.best[0],part.best[1],)
#     print part.best.fitness.values
# plt.xlim(-30,30)
# plt.ylim(-60,15)
# plt.show()
#quit()



with pm.Model() as model:
    
    params = pm.Flat('params', shape=(2))       
    #params =[0,1]
    #log_like = likelihood(model.x, model.y)
    like = pm.ArbLikelihood('like', likelihood(model.params))
    #like = pm.DensityDist('like', logp, observed=(log_F))
    #like = pm.Potential('like', likelihood(model.params))
    like = pm.Deterministic('like', likelihood(model.params))
    step = pm.Dream(blocked=True,start_random=False, save_history=False, parallel=True, history_file='ndim_banana_seed.npy')
    
    start = [{'params':centroids[chain]} for chain in range(nchains)]
    
    trace = pm.sample(10000, step, start=start, njobs=nchains,)
    #pm.traceplot(trace,vars=['params','like'])
    #plt.show()
    #plt.savefig('trace.png')
    #plt.clf()
    x1 = []
    x2 = []
    cm = plt.cm.get_cmap('RdYlBu')
    for i in range(nchains):
        colors = trace['like'][i][:]
        cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        plt.scatter(trace['params'][i][:,0], trace['params'][i][:,1],c=colors,cmap=cm,norm=cNorm)
        x1.append(trace['params'][i][:,0])
        x2.append(trace['params'][i][:,1])
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    x1= x1.flatten()
    x2= x2.flatten()
    plt.colorbar()
    plt.xlim(-30,30)
    plt.ylim(-60,15)
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.savefig('banana_dream_pso_10000_steps_5_chains.png')
    plt.show()
    plt.clf()
    hist,xedges,yedges = np.histogram2d(x1,x2,bins=40,range=np.array([(-30, 30), (-60.0, 15)]))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    plt.imshow(hist.T,origin='lower',interpolation='nearest',extent=extent)
    plt.colorbar()
    plt.title("PSO-DREAM sampling",fontsize=20,family='sans-serif')
    plt.xlabel('x1',fontsize=16,style='italic',family='sans-serif')
    plt.ylabel('x2',fontsize=16,style='italic',family='sans-serif')
    plt.savefig('pso_and_dream_only_2d_hist_5chains_10000_samples.png',dpi=150)
    plt.show()
    #dictionary_to_pickle = {}

    #for dictionary in trace:
    #    for var in dictionary:
    #       dictionary_to_pickle[var] = trace[var] 
    print pm.diagnostics.gelman_rubin(trace)
    #text.dump('2015_04_22_ndim_banana_mtdreamzs', trace)
    
    #pickle.dump(dictionary_to_pickle, open('2015_04_22_ndim_banana_mtdreamzs.p', 'wb'))

    
    



