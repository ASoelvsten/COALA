import mockdata
import mcmc
import numpy as np
import posteriorpredictive as postpre

# Basic settings

new_data = False    # Do you want to create your own mockdata (or do you use an existing file)
sims = 200000       # Number of simulations per chain in the mcmc
direc = "./MOCK02/" # Directory for storing output
burnin = 50000      # Number of samples to discard as burnin

#===================================================================

# Create new mock data

if new_data:
    mockdata.create_mockdata_fiji()

# Set priors

prior_low = -40.*np.ones(26)
prior_up = 40.*np.ones(26)

# For X
prior_low[1] = 0
prior_low[2] = 0
prior_low[3] = 0
prior_low[4] = 0
prior_low[5] = 0
prior_low[6] = 0
prior_low[7] = 0
prior_low[8] = 0
prior_low[10] = 0
prior_low[13] = 0
prior_low[14] = 0
prior_low[15] = 0
prior_low[16] = 0
prior_low[17] = 0

# For netowrks
prior_low[-8] = 0
prior_low[-6] = 0
prior_low[-7] = 0
prior_low[-5] = 0

# For information spread
prior_low[-3] = 0
prior_low[-2] = 0
prior_low[-1] = 0

prior_low[-4] = 0
prior_up[-4] = 1

# Run mcmc

mcmc.run_mcmc(prior_low,prior_up,sims=sims,direc=direc,mode="Fast_Fiji",disttype="heatmap",Restart=False)

# Run posterior predictive tests

# Introducting changes for counterfactual future scenario
changes1 = -1*np.ones(17)
changes1[5] = 1 # index -1 because we are counting indexes of X not of beta
#changes1[7] = 1
#changes1[4] = 1
changes2 = -1*np.ones(17)
changes2[5] = 0
#changes2[7] = 0
#changes2[4] = 0
changes = np.column_stack((changes1,changes2))
labels = ["Posterior predictive (sABM)", r"More champions $(\beta_6)$",r"No champions $(\beta_6)$"] # 5
#labels = ["Posterior predictive (sABM)", r"Hotels further away $(\beta_8)$",r"Hotels closer $(\beta_8)$"] # 7
#labels = ["Posterior predictive (sABM)", r"Higher overall benefit $(\beta_5)$",r"Lower overall benefit $(\beta_5)$"] # 7

sam = 100 # Number of samples drawn from posterior

only_measures = False
media = False # If you want to test impact of media

truth = np.load("truth.npy")

postpre.mcmc_sum(direc, "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=burnin,info=0,communicate_ne=False,communicate_ngo=False,communicate_q=False, communicate_base=True,Future=28,only_future=False,changes=changes,labels=labels,only_measures=False,truth=truth)
