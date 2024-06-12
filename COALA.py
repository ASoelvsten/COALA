############################################################
#               ____ ___    _    _        _                #
#              / ___/ _ \  / \  | |      / \               #
#             | |  | | | |/ _ \ | |     / _ \              #
#             | |__| |_| / ___ \| |___ / ___ \             #
#              \____\___/_/   \_\_____/_/   \_\            #
#                                                          #
#                     ___,        _,,_                     #
#                    /    \______/    \                    #
#                   /     '  __  '     \                   #
#                   \      o/  \o      /                   #
#                    \___|  |  |  |___/                    # 
#                        \  \__/  /                        #
#                         '______'                         #
#                                                          #
############################################################
# COnservation study ALgorithm using Agent-based modelling #
############################################################

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import time
import emcee
from multiprocessing import Pool
from collections import Counter
from scipy.stats import bernoulli, binom

#===========================================================
# Function for loading existing data sets.
#===========================================================

def init(mode = "Fiji"):

    if mode == "Fiji" or mode == "Fiji_0" or mode == "Fast_Fiji":

#        data = np.load("../FIJI/summary_fiji_XD.npy",allow_pickle=True)
        data = np.load("./summary_mock_XD.npy",allow_pickle=True)

        x = np.asarray(data[:,0],dtype=float)
        y = np.asarray(data[:,1],dtype=float)
        t = np.asarray(data[:,2],dtype=float)
        num_agents = len(x)

        colp = 24

        Island = data[:,3].copy()
        province = data[:,4].copy()
        district = data[:,5].copy()
        informed = data[:,6].copy()
        X = np.asarray(data[:,7:colp],dtype=float) # KEEP UP TO DATE
        distance_matrix = np.asarray(data[:,colp:colp+num_agents],dtype=float)
        DQ = data[:,colp+num_agents:colp+num_agents*2] # Qoliqoli
        DNGO = data[:,colp+num_agents*2:colp+num_agents*3]
        DC = data[:,colp+num_agents*3:colp+num_agents*4] # Chiefly
        if mode == "Fiji_0":
            DNGO = np.zeros(np.shape(DNGO))
        adag =  np.asarray(data[:,colp+num_agents*4:],dtype=float)

        AD = []
        ad = 0
        for i in set(t):
            if i != 0:
                ad += np.sum([t==i])/num_agents
                a = [i,ad]
                if AD == []:
                    AD = a.copy()
                else:
                    AD = np.column_stack((AD,a))

        agents = np.zeros(num_agents)

#        informed = np.ones(num_agents)

        return agents, informed, province, X, distance_matrix, AD, x , y, adag, DNGO, DQ, DC

    if mode == "Fiji2" or mode == "Fiji2_0" or mode == "Fast_Fiji2":

        data = np.load("../FIJI/summary_fiji_all_D.npy",allow_pickle=True)
#        data = np.load("./summary_mock_all_D.npy",allow_pickle=True)

        x = np.asarray(data[:,0].copy(),dtype=float)
        y = np.asarray(data[:,1].copy(),dtype=float)
        t = np.asarray(data[:,2].copy(),dtype=float)
        num_agents = len(x)

        informed = data[:,3].copy() # all members of NGOs
#        informed = np.ones(len(data[:,3]))
        X = []

        distance_matrix = np.asarray(data[:,4:4+num_agents],dtype=float) # KEEP 4
        DQ = np.asarray(data[:,4+num_agents:4+num_agents*2],dtype=float)
        DNGO = np.asarray(data[:,4+num_agents*2:4+num_agents*3],dtype=float)
        if mode == "Fiji2_0":
            DNGO = np.zeros(np.shape(DNGO))
        adag =  np.asarray(data[:,4+num_agents*3:],dtype=float)
        DC = np.zeros((num_agents, num_agents))

        AD = []
        ad = 0

        years = np.linspace(2000,2017,18)

        for i in years:
            if i != 0:
                ad += np.sum([t==i])/num_agents
                a = [i,ad]
                if AD == []:
                    AD = a.copy()
                else:
                    AD = np.column_stack((AD,a))

        agents = np.zeros(num_agents)

        return agents, informed, [], X, distance_matrix, AD, x , y, adag, DNGO, DQ, DC
 
    if mode == "Fiji3" or mode == "Fast_Fiji3":

        data = np.load("../FIJI/summary_fiji_all_XD.npy",allow_pickle=True)
#        data = np.load("./summary_mock_all_D.npy",allow_pickle=True)

        x = np.asarray(data[:,0],dtype=float)
        y = np.asarray(data[:,1],dtype=float)
        t = np.asarray(data[:,2],dtype=float)
        num_agents = len(x)

        informed = data[:,3] # all members of NGOs
#        informed = np.ones(len(data[:,3]))
        distance_matrix = np.asarray(data[:,4:4+num_agents],dtype=float) # KEEP 4
        DQ = np.asarray(data[:,4+num_agents:4+num_agents*2],dtype=float)
        DNGO = np.asarray(data[:,4+2*num_agents:4+num_agents*3],dtype=float)

        colp = 2

        adag =  np.asarray(data[:,4+num_agents*3:-colp],dtype=float) # KEEP UP TO DATE
        X =  np.asarray(data[:,-colp:],dtype=float)
        DC = np.zeros((num_agents, num_agents))

        AD = []
        ad = 0

        years = np.linspace(2000,2017,18)

        for i in years:
            if i != 0:
                ad += np.sum([t==i])/num_agents
                a = [i,ad]
                if AD == []:
                    AD = a.copy()
                else:
                    AD = np.column_stack((AD,a))

        agents = np.zeros(num_agents)

        return agents, informed, [], X, distance_matrix, AD, x , y, adag, DNGO, DQ, DC

#===========================================================
# Probability for single agent to adopt if informed for stochastic model.
#===========================================================

def p_adopt(i,pt,agents,informed,distance_matrix,X,ts,DNGO,DQ,DC,threshold=50):

    if informed[i]>0:
        mask = distance_matrix[i,:] == 1
        mask_NGO = DNGO[i,:] == 1
        mask_NQ = DQ[i,:] == 1
        mask_NC = DC[i,:] == 1
        N_ne = np.sum(agents[mask])
        N_ngo = np.sum(agents[mask_NGO])
        N_q = np.sum(agents[mask_NQ])
        N_c = np.sum(agents[mask_NC]) # Has your chief adopted?
        if X == []:
            logitp = pt[0]+pt[-4]*N_ne+pt[-3]*N_ngo+pt[-2]*N_q+pt[-1]*N_c
        else:    
            logitp = pt[0]+np.sum(pt[1:-4]*X[i,:])+pt[-4]*N_ne+pt[-3]*N_ngo+pt[-2]*N_q+pt[-1]*N_c

        p = np.exp(logitp)
        p = p/(1+p)

    else:

        p = 0

    return p

#===========================================================
# Dummy function for implementing abandoning of initiative.
# Currently not in use.
#===========================================================

def p_abandon():
    return 0.00

#===========================================================
# Function future predictions under currently contrafactual circumstances
#===========================================================

def future_change(X,distance_matrix,DNGO,changes):

    for i, c in enumerate(changes):
        if c == 0:
            X[:,i] = np.min(X[:,i])
            print(i, "set to", np.min(X[:,i]))
        if c == 1:
            X[:,i] = np.max(X[:,i])
            print(i, "set to", np.max(X[:,i]))
           
    return X, distance_matrix, DNGO


#===========================================================
# Stochastic agent-based model for conservation
#===========================================================

def run_coala(param,mode="Fiji",step=[],dt = 1.,communicate_ne=True,communicate_ngo=True,communicate_q=True,Future=0,changes=None):

    # Settings:

    agents, informed, interchangeable, X, distance_matrix, AD, x, y, adag, DNGO, DQ, DC = init(mode=mode)

    if mode == "Fiji" or mode == "Fiji_0":
        steps = int(np.max(AD[0,:])-np.min(AD[0,:])+2)
    elif  mode == "Fiji2" or mode == "Fiji2_0" or mode == "Fiji3" or mode == "Fast_Fiji3":
        steps = int(np.max(AD[0,:])-np.min(AD[0,:])+2)
    
    if Future !=0:
        past = steps-1
        steps = Future

    num_agents = len(agents)

    index = np.arange(num_agents)

    ADOPTERS = agents.copy()

    # Loop over time steps.

    t = [0]

    for ts in range(1,steps):
     
        t.extend([ts*dt])
    
        sind = sorted(index, key=lambda k: random.random())

        if Future !=0 and changes is not None and ts > past:
            if isinstance(changes, str) and changes == "Media":
                param[-1] = 0
            else:
                X, distance_matrix, DNGO = future_change(X,distance_matrix,DNGO,changes)
                changes = None               

        # Loop over agents (in random order)

        for i in sind:

            # Agents might randomly hear about the initiative, e.g. on television or by visiting far of communities.
            if informed[i]==0:
                pdummy = np.random.uniform(0,1)
                if pdummy < param[-1]:
                    informed[i] = 1

            if agents[i] == 0 and informed[i]>0:
                adopt = p_adopt(i,param[:-1].copy(),agents,informed,distance_matrix,X,t[ts],DNGO,DQ,DC)
                pdummy = np.random.uniform(0,1)
                if pdummy < adopt*dt: # Does dt*p make sense beyond a toy model?
                    agents[i] = 1. # agent adopts
                    if communicate_ne:
                        informed[distance_matrix[i,:] == 1] = 1 # all neighbours thereby get to know about initiative
                    if communicate_ngo:
                        informed[DNGO[i,:] == 1] = 1 # all members of your NGO get to know about initiative
                    if interchangeable != []:
                        informed[interchangeable == interchangeable[i]] = 1 # all neighbours in your district get to know too
                    if communicate_q:
                        informed[DQ[i,:] == 1] = 1 # Tell your last network, qoliqolis

#            if agents[i] == 1:
#                abandon = p_abandon()
#                pdummy = np.random.uniform(0,1)
#                if pdummy < abandon*dt:
#                    agents[i] = -1

        ADOPTERS = np.vstack((ADOPTERS,agents))
      
    return ADOPTERS, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, informed

#===========================================================
# Deterministic version of COALA
#===========================================================

def fast_coala(param,mode="Fiji",step=[],dt = 1.):

    agents, informed, interchangeable, X, distance_matrix, AD, x, y, adag, DNGO, DQ, DC = init(mode=mode)

    if mode == "Fiji" or mode == "Fiji_0" or mode =="Fast_Fiji":
        steps = int(np.max(AD[0,:])-np.min(AD[0,:])+2)
    elif  mode == "Fiji2" or mode == "Fiji2_0" or mode == "Fast_Fiji2" or mode == "Fiji3" or mode == "Fast_Fiji3":
        steps = int(np.max(AD[0,:])-np.min(AD[0,:])+2)
    
    num_agents = len(agents)

    P = agents.copy()
    
    I = informed.copy()

    t = [0]
    
    for ts in range(1,steps):
         
        t.extend([ts*dt])
      
        if ts == 1:
            p0 = np.asarray(P.copy(),dtype=float)
            i0 = np.asarray(I.copy(),dtype=float)
        else:
            p0 = P[:,-1].copy()
            i0 = I[:,-1].copy()
        
        # We need to include the spread of information. This is assumed to be a truely random process.
        # However, we want to avoid actual randomness so as not to run into stochastic behaviour when sampling.
        # Moreover, one assumption in the ABM is that you are informed if someone in your network adopted or randomly.
        # We can put it directly into the equation:
        
        # P(adopt & informed) = P(adopt|informed)P(informed)
               
        # We just run through all agents at the same time. 
        # The implicit assumption is that you might be affected by your neighbours 
        # but not if they have just adopted within the same year.

        N_ngo = np.sum(np.multiply(p0, DNGO),axis=0)
    
        N_ne = np.sum(np.multiply(p0, distance_matrix),axis=0)

        N_q = np.sum(np.multiply(p0, DQ),axis=0)

        N_c = np.sum(np.multiply(p0, DC),axis=0) # Has your chief adopted?
               
        if X == []:
            logitp = param[0]+param[-8]*N_ne+param[-7]*N_ngo+param[-6]*N_q+param[-5]*N_c
        else:
            logitp = param[0]+np.sum(np.multiply(param[1:-8],X),axis=1)+param[-8]*N_ne+param[-7]*N_ngo+param[-6]*N_q+param[-5]*N_c
    
#        logitp = np.array(logitp, dtype=np.float128) 
        p = np.exp(logitp.astype(float))
        p = p/(1+p)
        p[np.logical_and(np.isnan(p),logitp>0)] = 1   # Due to numerical issues p might become nan
                
        # Probability of being informed. this must arguably depend on random spread of information and
        # the number of neighbours who have already adopted.
        
        BR = -8-np.log(num_agents)

        logitI = BR+param[-3]*N_ne+param[-2]*N_ngo+param[-1]*N_q
#        logitI = np.array(logitp, dtype=np.float128)
       
        i = np.exp(logitI.astype(float))
        i = i/(1+i) + param[-4]
        i[np.logical_and(np.isnan(i),logitI>0)] = 1         
        i[i>1] = 1 

        # Up until now, i is the probability to be informed in the current time step. But we want
        # to know the probability of being informed at this time step.  

        i = i0 + (1-i0)*i
        
        p = p*i

        # Up until now, we have looked at the probability for adoption within this time step.
        # However, what we want to store is the probability of adopting up until now.
    
        p = p0 + (1-p0)*p # Probability of adoption

        P = np.column_stack((P,p))
        I = np.column_stack((I,i))

    return P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I

#===========================================================
# The functions below are all for the evaluation of the output,
# i.e. for comparison between data and model predictions.
#===========================================================

def similarity(adopters,interchangeable):

    dis = sorted(set(interchangeable))
    COUNTS = []
    for ti in range(np.shape(adopters)[0]):
        counts = []
        real_adopters = interchangeable[np.asarray(adopters[ti,:],dtype=bool)]
        count = Counter(real_adopters)
        for di in dis:
            counts.extend([count[di]])
        if ti == 0:
            COUNTS = counts.copy()
        else:
            COUNTS = np.vstack((COUNTS,counts))

    return COUNTS

#===========================================================
# Function defining different measures to evaluate the goodness of fit.
#===========================================================

def distance(repeats,param,disttype="district",profile=False,verbose=False,mode="Fiji",communicate_ne=True,communicate_ngo=True,communicate_q=True,Future=0,changes=None):

    # Comparison agent by agent

    if disttype == "heatmap1" or disttype == "heatmap2":
        for i in range(repeats):
            ADOPTERS, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable,informed = run_coala(param,mode=mode, communicate_ne=communicate_ne, communicate_ngo=communicate_ngo, Future=Future,communicate_q=communicate_q, changes=changes)
            if i == 0:
                temp = ADOPTERS.copy()
            else:
                temp = temp + ADOPTERS
        temp /= repeats

        if disttype == "heatmap1" and Future==0:
            d = np.size(adag)-(np.sum(np.multiply(temp[1:,:].T,adag))+np.sum(np.multiply(1-np.asarray(temp[1:,:].T),1-np.asarray(adag))))
        elif Future == 0:
            d = np.sqrt(np.sum((temp[1:,:].T-adag)**2))
        else:
            d = 0

    # F-score

    if disttype == "F1":
        F1 = -1.
        for i in range(repeats):
            ADOPTERS, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable,informed = run_coala(param,mode=mode,communicate_ne=communicate_ne, communicate_ngo=communicate_ngo, communicate_q=communicate_q, Future=Future, changes = changes)
            acad = adag.T
            prad = ADOPTERS[1:,:]
            TP = np.sum(np.logical_and(acad == 1,prad==1)) # True positives
            FP = np.sum(np.logical_and(acad == 0,prad==1)) # False positives
            FN = np.sum(np.logical_and(acad == 1,prad==0)) # False negatives
            TN = np.sum(np.logical_and(acad == 0,prad==0)) # True negatives
            f1 = TP/(TP+0.5*(FP+FN))
            if f1 > F1:
                F1 = f1
                temp = ADOPTERS.copy()
        d = 1 - F1

    res = 0

    # Compare total number of adopters as a function of time only

    if disttype == "SumAd":
        RES = []
        for i in range(repeats):
            ADOPTERS, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable,informed = run_coala(param,mode=mode,communicate_ne=communicate_ne, communicate_ngo=communicate_ngo, communicate_q=communicate_q, Future=Future, changes = changes)
            res = 100*np.sum(ADOPTERS[:,:],axis=1)/len(adag)
            if i == 0:
                RES = res.copy()
            else:
                RES = np.column_stack((RES,res))
        
        d = np.sqrt(np.sum((np.mean(RES[1:,:],axis=1)-100*AD[1,:])**2/np.var(RES[1:,:],axis=1)))

        res = RES.copy()
        profile = False

    if profile:
        res = 100*np.sum(temp[:,:],axis=1)/len(adag)

    # Define how much information we want to return.

    if verbose:
        return d, res, temp, t, adag, AD, distance_matrix, x, y
    else:
        return d, res

#===========================================================
# Log-likelihood functions for the deterministic model.
#===========================================================

def d_fast(param,mode="Fast_Fiji",disttype="heatmap"):

    P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = fast_coala(param,mode=mode)
    
    P = P.astype(float)

    if disttype=="heatmap": #mode == "Fast_Fji" or mode == "Fast_Fiji3":    
        L = np.sum(np.log(P[:,1:][adag==1]))+np.sum(np.log(1-np.asarray(P[:,1:][adag==0])))
        L = 2*np.sqrt(-L)
    elif disttype=="Curve":
        L = 0
        for i in range(np.shape(adag)[1]):
            # Given that we only have a curve... now it becomes binomial
            A = np.array(adag[:,i], dtype=float)
            B = np.array(P[:,1+i], dtype=float)
            prob = binom.logpmf(np.sum(A), len(A), np.mean(B))
            L += prob
        L = 2*np.sqrt(-L)

    return L

#===========================================================
# Wrapping function to call log-likelihood. 
#===========================================================

def fiji_distance(param,repeats=100,disttype="F1",profile=True,mode="Fiji"):
    if mode == "Fast_Fiji" or mode == "Fast_Fiji2" or mode == "Fast_Fiji3":
        d = d_fast(param,mode=mode,disttype=disttype)
    else:
        d, res = distance(repeats, param, disttype=disttype,profile=profile,mode=mode)
    return d
