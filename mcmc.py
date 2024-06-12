import numpy as np
import COALA as coala
from time import time
import matplotlib.pyplot as plt
import emcee # For MCMC
from multiprocessing import Pool
import pygtc
from scipy.stats import bernoulli, binom
import os

# This script includes the functions needed to run the MCMC. For more information, we
# refer to the documentation of emcee: https://emcee.readthedocs.io/en/stable/

#============================================================================
# Defining priors
#============================================================================

def get_prior(params,prior_low,prior_up,mode):

    p = 0

    if type(prior_low) != type(1.0) and type(prior_up) != type(1.0):
        if any(params < prior_low[:len(params)]) or any(params > prior_up[:len(params)]):
            p = -np.inf
    else:
        if any(params < prior_low) or any(params > prior_up):
            p = -np.inf

    return p

#============================================================================
# Defining the log-likelihood function
#============================================================================

def likelihood(param,prior_low,prior_up,mode,disttype):

    prior = get_prior(param,prior_low,prior_up,mode)

    if prior != -np.inf:

#        if mode == "Fast_Fiji":
#            param = np.insert(param,0, -6)

        P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = coala.fast_coala(param,mode=mode)

        P = P.astype(float)

        if disttype == "heatmap": #mode == "Fast_Fiji" or mode == "Fast_Fiji3" or disttype=="heatmap":
            # We random events binomial or Bernouilli. The entries in p are the probability for the event occuring.
            # So, if we observe the event, it should have been there with probability p. If we don't observe it, the
            # probability for this is (1-p).
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html
            L = np.sum(np.log(P[:,1:][adag==1]))+np.sum(np.log(1-np.asarray(P[:,1:][adag==0])))

        elif disttype == "heatmap_L2":
            L = -np.sum((P[:,1:]-adag)**2)

        elif disttype == "Curve":
            L = 0
            for i in range(9):
                # Given that we only have a curve... now it becomes binomial
                A = np.array(adag[:,i], dtype=np.float64)
                B = np.array(P[:,1+i], dtype=np.float64)
                prob = binom.logpmf(np.sum(A), len(A), np.mean(B))
                L += prob 

        elif disttype == "Curve_L2":
            res = 100*np.sum(P,axis=0)/len(adag)
            L = -np.sum((res[1:]-100*np.sum(adag,axis=0)/len(adag))**2)

        print(L)

    else:

        L = -np.inf

    return L

#============================================================================
# When initialising the runs, we randomly set the initial conditions. Here,
# we ensure that the initial conditions lie within the priors.
#============================================================================

def prune_p0(p0,prior_low,prior_up):

    rows = np.shape(p0)[0]
    col = np.shape(p0)[1]
    for r in range(rows):
        for c in range(col):
            if p0[r,c] > prior_up[c] and -p0[r,c] < prior_low[c]:
                p0[r,c] = prior_up[c]
                print(r,c,"Set to upper boundary")
            elif p0[r,c] < prior_low[c] and -p0[r,c] > prior_up[c]:
                p0[r,c] = prior_low[c]
                print(r,c,"Set to lower boundary")
            elif p0[r,c] > prior_up[c] :
                p0[r,c] = -p0[r,c]
            elif p0[r,c] < prior_low[c]:
                p0[r,c] = -p0[r,c]

    return p0

#============================================================================
# Run the MCMC.
#============================================================================

def run_mcmc(prior_low,prior_up,sims = 100,mode="Fast_Fiji",direc="./",disttype="heatmap",Restart=False):

    # Settings

    if mode == "Fast_Fiji":
        ndim = 26
    elif mode == "Fast_Fiji2":
        ndim = 9
    elif mode == "Fast_Fiji3":
        ndim = 11

    nwalkers = 2*ndim

    low = np.zeros(ndim)
    low[0] = -7

    scale = np.ones(ndim)*0.5
    scale[-7] = 0.1
    scale[-6] = 0.1
    scale[-8] = 0.1

    # Initial conditions

    if Restart:
        sampleso = np.load(direc+"/samples.npy")
        p0 = sampleso[-nwalkers:,:]
        logpo = np.load(direc+"/lnL.npy")
        if os.path.isfile(direc+"/cl.npy"):
            clo = np.load(direc+"/cl.npy")
    else:
        p0 = np.random.normal(loc=low,scale=scale,size=(nwalkers,ndim))
        print(np.sign(prior_low),p0)
        p0 = prune_p0(p0,prior_low,prior_up)


    start = time()
    with Pool() as pool:
        print("Sampling with %i walkers using multiple cores ..." %(nwalkers))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=[prior_low,prior_up,mode,disttype], pool=pool)
        sampler.run_mcmc(p0, sims)
        print("Sampling finished.")
        samples = sampler.get_chain(flat=True)
        print("Saving results ...")
        if Restart:
            samples = np.vstack((sampleso,samples))
        np.save(direc+"/samples.npy",samples)
        print("Re-computing posteriors for all samples ...")
        logp = sampler.get_log_prob(flat=True)
        if Restart:
            logp = np.asarray(list(logpo)+list(logp))
        np.save(direc+"/lnL.npy",logp)
        print(time()-start)
#        cl = sampler.get_autocorr_time()
#        print("Autocorrelation times: ", cl)
#        if Restart and os.path.isfile(direc+"/cl.npy"):
#            cl = np.vstack((clo,cl))
#        np.save(direc+"/cl.npy")
        print("Done.")
