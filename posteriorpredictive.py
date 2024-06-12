import numpy as np
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import time
import COALA as coala
import pygtc
from glob import glob
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pygtc
import mcmc

# This script is used to perform the posterior predictive tests and plots presented in the paper.

#===========================================================
# Run model for a number of parameter sets including time points
# in the future, i.e., outside of the data set.
#===========================================================

def sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,changes,label,info=0):

    RES = []

    for i in index:
        ppt = samples[i,:-3].copy()
        if not communicate_base:
            print(ppt[-1])
            ppt[-1] = info*ppt[-1] 
        d, res, ADOPTERS, t, adag, AD, distance_matrix, x, y = coala.distance(1,ppt,disttype="heatmap1",profile=True,verbose=True,mode=mode2, communicate_ne=communicate_ne,communicate_ngo=communicate_ngo,communicate_q=communicate_q,Future=Future,changes=changes)
        print(res)
        if RES == []:
            RES = res.copy()
        else:
            RES = np.column_stack((RES,res))

    np.save(direc+"/res_"+label+".npy",RES)

    m = np.mean(RES,axis=1)
    s = np.std(RES,axis=1)

    return m, s, t, AD

#===========================================================
# Run scenarios with different types of information spread.
#===========================================================

def future_media(direc, samples, mode , Future, s3, communicate_ne,communicate_ngo,communicate_q,communicate_base,labels,info=0,sparse=False):

    index = np.arange(len(samples))
    if len(samples) > s3:
        index = np.random.choice(index,s3)

    print("Sampling models ...")

    if mode == "Fast_Fiji":
        mode2 = "Fiji"
    else:
        mode2 = "Fiji2"

    # Base
    m, s, t, AD =  sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,None,labels[0],info=0)
    # No media at all
    mc, sc, _ , _ = sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,False,None,labels[1],info=0)
    m = np.column_stack((m,mc))
    s = np.column_stack((s,sc))
    # Only media
    mc, sc, _ , _ = sample_future(direc,index,samples, mode2, Future,False,False,False,communicate_base,None,labels[1],info=0)
    m = np.column_stack((m,mc))
    s = np.column_stack((s,sc))
    # No media in future
    mc, sc, _ , _ = sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,"Media",labels[2],info=0)
    m = np.column_stack((m,mc))
    s = np.column_stack((s,sc))

    plt.figure(figsize=(10,8))
    plt.plot(AD[0,:],100*AD[1,:],"ko",label="Data")

    colors = ["orange","green","orangered","blue","purble","blue"]
    linestyles = ["-","-.","--",":"]

    for i in range(np.shape(m)[1]):
        if i > 2:
            steps = np.shape(AD)[1]+1
        else:
            steps = 0
        plt.fill_between(x=np.arange(Future-steps)+np.min(AD[0,:])+steps-1,y1=m[steps:,i]-s[steps:,i],y2=m[steps:,i]+s[steps:,i],color=colors[i],alpha=0.2)
        plt.plot(np.arange(Future-steps)+np.min(AD[0,:])+steps-1,m[steps:,i],linestyle=linestyles[i],color=colors[i],label=labels[i])

    plt.axvline(x=+np.min(AD[0,:])+t[np.shape(AD)[1]],color="grey",linestyle="--")
    plt.xlabel("Time [years]",fontsize=14)
    plt.ylabel("Fraction of adopters [%]",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14,loc="best")

    np.save(direc+"/Future_pred_media.npy",np.column_stack((np.arange(Future)+np.min(AD[0,:]),m,s))) 

    if sparse:
        n1 = "/postpred_future_media_sparse.jpeg"
    else:
        n1 = "/postpred_future_media.jpeg"

    plt.savefig(direc+n1,bbox_inches='tight')

#===========================================================
#
#===========================================================

def future_coala(direc, samples, mode , Future, s3, communicate_ne,communicate_ngo,communicate_q,communicate_base,changes,labels,info=0,sparse=False):

    index = np.arange(len(samples))
    if len(samples) > s3:
        index = np.random.choice(index,s3)

    print("Sampling models ...")

    if mode == "Fast_Fiji":
        mode2 = "Fiji"
    else:
        mode2 = "Fiji2"

    m, s, t, AD =  sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,None,labels[0],info=info)

    if changes is not None:
     
        if changes.ndim == 2:
            for i in range(np.shape(changes)[1]):
                mc, sc, _ , _ = sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,changes[:,i],labels[i+1],info=info)
                m = np.column_stack((m,mc))
                s = np.column_stack((s,sc))
        else:
             mc, sc, _, _ = sample_future(direc,index,samples, mode2, Future,communicate_ne,communicate_ngo,communicate_q,communicate_base,changes,labels[1],info=info)
             m = np.column_stack((m,mc))
             s = np.column_stack((s,sc))

    plt.figure(figsize=(10,8))
    plt.plot(AD[0,:],100*AD[1,:],"ko",label="Data")
    
    colors = ["orange","green","magenta","purple","cyan","blue"]
    linestyles = ["-","-.",":"]

    if changes is None:

        plt.fill_between(x=np.arange(Future)+np.min(AD[0,:]),y1=m-s,y2=m+s,color="orange",alpha=0.2)
        plt.plot(np.arange(Future)+np.min(AD[0,:]),m,color="orange",label="Posterior predictive (sABM)")

    else:
        for i in range(np.shape(m)[1]):
            if i > 0:
                steps = np.shape(AD)[1]+1
            else:
                steps = 0
            plt.fill_between(x=np.arange(Future-steps)+np.min(AD[0,:])+steps-1,y1=m[steps:,i]-s[steps:,i],y2=m[steps:,i]+s[steps:,i],color=colors[i],alpha=0.2)
            plt.plot(np.arange(Future-steps)+np.min(AD[0,:])+steps-1,m[steps:,i],linestyle=linestyles[i],color=colors[i],label=labels[i])

    plt.axvline(x=+np.min(AD[0,:])+t[np.shape(AD)[1]],color="grey",linestyle="--")
    plt.xlabel("Time [years]",fontsize=14)
    plt.ylabel("Fraction of adopters [%]",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14,loc="best")

    np.save(direc+"/Future_pred.npy",np.column_stack((np.arange(Future)+np.min(AD[0,:]),m,s))) 

    if sparse:
        n1 = "/postpred_future_sparse.jpeg"
    else:
        n1 = "/postpred_future.jpeg"

    plt.savefig(direc+n1,bbox_inches='tight')

#===========================================================
# Create summary of posterior distributions. 
#===========================================================

def forest_plot(direc,samples,logp,names,burnin, mode, samples_fine = [],sparse=False,only_measures=False,truth=[]):

    fig, ax = plt.subplots(figsize=(10, 10))
    results = samples[burnin:,:].copy()
    results[:,-1] = results[:,-1]/10
    results[:,-2] = results[:,-2]/10
    results[:,-3] = results[:,-3]/10
#    results[:,-5] = results[:,-5]/10
#    results[:,-6] = results[:,-6]/10
    if samples_fine != []:
        results = samples_fine.copy()
        results[:,-1] = results[:,-1]/10
        results[:,-2] = results[:,-2]/10
        results[:,-3] = results[:,-3]/10
#        results[:,-5] = results[:,-5]/10
#        results[:,-6] = results[:,-6]/10

    names1 = names.copy()

    Med = np.median(results,axis=0) # Note burnin already out
    print("logp",logp[logp.argmax()])
    Best = samples[np.nanargmax(logp),:]

    print("BEST", Best)

    if not only_measures:

        p025 = np.percentile(results,2.5,axis=0)
        p975 = np.percentile(results,97.5,axis=0)

        summary = np.vstack((Best,Med,p025,p975))

        np.save(direc+"/summary.npy",summary)

        if mode == "Fast_Fiji2" or mode == "Fast_Fiji3":
            Med = np.delete(Med,-5)
            p025 = np.delete(p025,-5)
            p975 = np.delete(p975,-5)
            names1 = np.delete(names1,-5)

        xerr = np.vstack((Med-p025,p975-Med))

        y = np.arange(samples.shape[1])

        print(Best, y)

        BB = Best.copy()

#        BB[-5] = BB[-5]/10
#        BB[-6] = BB[-6]/10

        BB[-1] = BB[-1]/10
        BB[-2] = BB[-2]/10
        BB[-3] = BB[-3]/10

        if mode == "Fast_Fiji2" or mode == "Fast_Fiji3":
            BB = np.delete(BB,-5)
            y = np.delete(y,-1)

        plt.errorbar(np.asarray(Med),y,xerr=xerr,fmt="o",capsize=2,capthick=2, alpha=0.7,label="Distribution",lw=3)
        plt.axvline(x=0,linestyle="--",color="grey")
        plt.plot(np.asarray(BB),y,"^",label="Best",markersize=12,lw=3)
        if truth != []:
            plt.plot(np.asarray(truth),y,"s",label="Truth",markersize=12,lw=3)
        plt.yticks(y,names1,size=16)
        plt.legend(fontsize=16)
        plt.gca().invert_yaxis()
        plt.xlabel(r"$\mathrm{Parameter\,\,values}$",fontsize=16)
        plt.ylabel(r"$\mathrm{Parameters}$",fontsize=16)
        plt.xticks(size=16)
        plt.yticks(size=16)

        if sparse:
            n1 = "/summary_sparse.jpeg"
        else:
            n1 = "/summary.jpeg"

        plt.savefig(direc+n1,bbox_inches='tight')

    P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, IB = coala.fast_coala(Best,mode=mode)

    return Best, IB

#===========================================================
# Perform posterior predictive test storing various performance metrics
#===========================================================

def postpred(direc,best_param,mode,samples,s1,s2,Best, IB,info,communicate_ne,communicate_ngo,communicate_q,communicate_base,sparse=False,only_measures=False):

    P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = coala.fast_coala(best_param,mode=mode)
    res_fast_med = 100*np.sum(P,axis=0)/len(adag)

    index = np.arange(len(samples))
    if len(samples) > s1:
        index = np.random.choice(index,s1)

    print(index, len(index),s1)

    RESd = []

    L1 = []
    L2 = []
    L3 = []
    L4 = []
    L5 = []
    L6 = []
    PA = []
    PNA = []
    PAMI = []
    PAMA = []
    PNAMA = []
    PNAMI = []

    P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = coala.fast_coala(Best,mode=mode)

    BL1 = mcmc.likelihood(Best,-100*np.ones(len(Best)),100*np.ones(len(Best)),mode,"heatmap")
    BL2 = mcmc.likelihood(Best,-100*np.ones(len(Best)),100*np.ones(len(Best)),mode,"Curve")
    BL3 = mcmc.likelihood(Best,-100*np.ones(len(Best)),100*np.ones(len(Best)),mode,"heatmap_L2")
    BL4 = mcmc.likelihood(Best,-100*np.ones(len(Best)),100*np.ones(len(Best)),mode,"Curve_L2")
    mae = 100*np.sum(P,axis=0)/len(adag)
    mae = np.mean(abs(mae[1:]-100*np.sum(adag,axis=0)/len(adag)))
    BL5 = mae
    maes = np.mean(abs(P[:,1:]-adag))
    BL6 = maes

    print("BEST Measures:", BL1, BL2, BL3, BL4, BL5, BL6)

    print("Sampling models ...")

    for i in index:
        ppt = samples[i,:].copy()
        print(i)
        P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = coala.fast_coala(ppt,mode=mode)

        pa =  np.median(P[:,1:][adag==1])
        pami =  np.min(P[:,1:][adag==1])
        pama =  np.max(P[:,1:][adag==1])
        pna = np.median(P[:,1:][adag==0])
        pnami = np.min(P[:,1:][adag==0])
        pnama = np.max(P[:,1:][adag==0])

        print("parameters:", pa,pami,pama,pna,pnami,pnama)
        PA.extend([pa])
        PNA.extend([pna])
        PAMI.extend([pami])
        PNAMI.extend([pnami])
        PAMA.extend([pama])
        PNAMA.extend([pnama])

        L1.extend([mcmc.likelihood(ppt,-100*np.ones(len(ppt)),100*np.ones(len(ppt)),mode,"heatmap")])
        L2.extend([mcmc.likelihood(ppt,-100*np.ones(len(ppt)),100*np.ones(len(ppt)),mode,"Curve")])
        L3.extend([mcmc.likelihood(ppt,-100*np.ones(len(ppt)),100*np.ones(len(ppt)),mode,"heatmap_L2")])
        L4.extend([mcmc.likelihood(ppt,-100*np.ones(len(ppt)),100*np.ones(len(ppt)),mode,"Curve_L2")])
        mae = 100*np.sum(P,axis=0)/len(adag)
        mae = np.mean(abs(mae[1:]-100*np.sum(adag,axis=0)/len(adag)))
        L5.extend([mae])
        maes = np.mean(abs(P[:,1:]-adag))
        L6.extend([maes])

        print("Measures:", L1[-1], L2[-1], L3[-1], L4[-1], L5[-1], L6[-1])

        res = 100*np.sum(P,axis=0)/len(adag)
        if RESd == []:
            RESd = res.copy()
        else:
            RESd = np.column_stack((RESd,res))

    if sparse:
        ext = "_S"
    else:
        ext=""

    np.save(direc+"/Measures"+ext+".npy",np.column_stack((L1,L2,L3,L4,L5,L6,PA,PAMI,PAMA,PNA,PNAMI,PNAMA)))

    RESd = RESd.astype(float)
    m = np.mean(RESd,axis=1)
    s = np.std(RESd,axis=1)

    plt.figure(figsize=(10,8))
    plt.plot(AD[0,:],100*AD[1,:],"ko",label="Data")
    plt.fill_between(x=t+np.min(AD[0,:])-1,y1=m-s,y2=m+s,color="firebrick",alpha=0.2)
    plt.plot(t+np.min(AD[0,:])-1,m,color="firebrick",label="Posterior predictive (dABM)")

    index = np.arange(len(samples))
    if len(samples) > s2:
        index = np.random.choice(index,s2)

    print("Sampling models ...")

    if mode == "Fast_Fiji":
        mode2 = "Fiji"
    else:
        mode2 = "Fiji2"

    RES2 = []
    HM = []
    D = 1e8

    TP = []
    FP = []
    FN = []
    TN = []
    TPR = []
    FPR = []
    FNR = []
    TNR = []

    for i in index:
        ppt = samples[i,:-3].copy()
        if not communicate_base:
            ppt[-1] = info*ppt[-1] 
            print(ppt[-1])
        d, res, ADOPTERS, t, adag, AD, distance_matrix, x, y = coala.distance(1,ppt,disttype="heatmap1",profile=True,verbose=True,mode=mode2,communicate_ne=communicate_ne,communicate_ngo=communicate_ngo,communicate_q=communicate_q,Future=0)
        print("Res:", res)

        acad = adag.T
        prad = ADOPTERS[1:,:]
        TP.extend([np.sum(np.logical_and(acad == 1,prad==1))])
        FP.extend([np.sum(np.logical_and(acad == 0,prad==1))])
        FN.extend([np.sum(np.logical_and(acad == 1,prad==0))])
        TN.extend([np.sum(np.logical_and(acad == 0,prad==0))])
        TPR.extend([np.sum(np.logical_and(acad == 1,prad==1))/np.sum(acad==1)])
        FPR.extend([np.sum(np.logical_and(acad == 0,prad==1))/np.sum(acad==0)])
        FNR.extend([np.sum(np.logical_and(acad == 1,prad==0))/np.sum(acad==1)])
        TNR.extend([np.sum(np.logical_and(acad == 0,prad==0))/np.sum(acad==0)])

        if RES2 == []:
            RES2 = res.copy()
            HM = ADOPTERS.copy()
        else:
            RES2 = np.column_stack((RES2,res))
            HM = HM + ADOPTERS
        if d < D:
            best = res.copy()
            best_last = ADOPTERS.copy()
            D = d

    F1 = np.asarray(TP)/(np.asarray(TP)+0.5*(np.asarray(FP)+np.asarray(FN)))

    np.save(direc+"/Confusion_matrix"+ext+".npy",np.column_stack((TPR,FPR,FNR,TNR,F1,TP,FP,FN,TN)))

    if not only_measures:

        HM /= s2

        m2 = np.mean(RES2,axis=1)
        s2 = np.std(RES2,axis=1)

        plt.fill_between(x=t+np.min(AD[0,:])-1,y1=m2-s2,y2=m2+s2,color="orange",alpha=0.2)
        plt.plot(t+np.min(AD[0,:])-1,m2,color="orange",label="Posterior predictive (sABM)")
    
        P, distance_matrix, AD, t, num_agents, x, y, adag, interchangeable, I = coala.fast_coala(Best,mode=mode)
        res_best = 100*np.sum(P,axis=0)/len(adag)

        plt.plot(t+np.min(AD[0,:])-1,res_best,"--",label="Best (dABM)",color="purple",dashes=(5, 5))

        plt.xlabel("Time [years]",fontsize=14)
        plt.ylabel("Fraction of adopters [%]",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)
        if sparse:
            n1 = "/postpred_sparse.jpeg"
        else:
            n1 = "/postpred.jpeg"

        plt.savefig(direc+n1,bbox_inches='tight')

        return RESd, RES2, HM, P, x, y, adag, AD, best, best_last

    else:

        return [],[],[],[],[],[],[],[],[],[]

#===========================================================
# Plot the spatial spread
#===========================================================

def mapf(direc,HM,P, x,y,adag,AD,best,best_last,sparse=False):

    ysca = 410
    xsca = 460
    plt.figure()
    amap = plt.tricontourf(x,y,100*HM[-1,:], 1000, cmap="plasma",vmin=0,vmax=100)
    plt.close()

    for i in range(9):
        gs = gridspec.GridSpec(1,4,width_ratios=[10,0.5,0.1,10])
        fig = plt.figure(figsize=(25,10))
        ax1 = fig.add_subplot(gs[0])
        mask = best_last[i+1,:] == 1
        ax1.set_title("Simulated data "+str(int(AD[0,i])),fontsize=20)
        x0 = x.copy()
        y0 = y.copy()
        c0 = 100*HM[i+1,:].copy()
        ax1.imshow(mpimg.imread('Map-of-Fiji-Islands.png'),zorder=1)
        ax1.scatter(20+xsca*(x0-np.min(x))/(np.max(x)-np.min(x)),30+ysca-ysca*(y0-np.min(y))/(np.max(y)-np.min(y)),c=c0,s=2*c0+20,zorder=4,vmin=0,vmax=100.,cmap="plasma",marker="o")
        plt.xticks(size=16)
        plt.yticks(size=16)
        cbar = fig.colorbar(amap, cax=fig.add_subplot(gs[1]))
        cbar.set_label('Occurrence, percentage of repeats', rotation=90,fontsize=18,labelpad=10) #,labelpad=-100)
        cbar.ax.tick_params(labelsize=16)
        ax2 = fig.add_subplot(gs[3])
        x0 = x[adag[:,i]==1].copy()
        y0 = y[adag[:,i]==1].copy()
        ax2.set_title("Real data "+str(int(AD[0,i])),fontsize=20)
        ax2.imshow(mpimg.imread('Map-of-Fiji-Islands.png'),zorder=1)
        ax2.plot(20+xsca*(x0-np.min(x))/(np.max(x)-np.min(x)),30+ysca-ysca*(y0-np.min(y))/(np.max(y)-np.min(y)),"ro",zorder=2)
        plt.xticks(size=16)
        plt.yticks(size=16)

        if sparse:
            n1 = "_sparse.jpeg"
        else:
            n1 = ".jpeg"

        plt.savefig(direc+"/Combo_data_"+str(i)+n1,bbox_inches='tight')

    P = np.array(P, dtype=np.float64)

    HM = P.transpose()

    plt.figure()
    amap = plt.tricontourf(x,y,100*HM[-1,:], 1000, cmap="plasma",vmin=0,vmax=100)
    plt.close()

    for i in range(9):
        gs = gridspec.GridSpec(1,4,width_ratios=[10,0.5,0.1,10])
        fig = plt.figure(figsize=(25,10))
        ax1 = fig.add_subplot(gs[0])
        mask = best_last[i+1,:] == 1
        ax1.set_title("Simulated data "+str(int(AD[0,i])),fontsize=20)
        x0 = x.copy()
        y0 = y.copy()
        c0 = 100*HM[i+1,:].copy()
        ax1.imshow(mpimg.imread('Map-of-Fiji-Islands.png'),zorder=1)
        ax1.scatter(20+xsca*(x0-np.min(x))/(np.max(x)-np.min(x)),30+ysca-ysca*(y0-np.min(y))/(np.max(y)-np.min(y)),c=c0,s=2*c0+20,zorder=4,vmin=0,vmax=100.,cmap="plasma",marker="o")
        plt.xticks(size=16)
        plt.yticks(size=16)
        cbar = fig.colorbar(amap, cax=fig.add_subplot(gs[1]))
        cbar.set_label('Occurrence, percentage of repeats', rotation=90,fontsize=18,labelpad=10) #,labelpad=-100)
        cbar.ax.tick_params(labelsize=16)
        ax2 = fig.add_subplot(gs[3])
        x0 = x[adag[:,i]==1].copy()
        y0 = y[adag[:,i]==1].copy()
        ax2.set_title("Real data "+str(int(AD[0,i])),fontsize=20)
        ax2.imshow(mpimg.imread('Map-of-Fiji-Islands.png'),zorder=1)
        ax2.plot(20+xsca*(x0-np.min(x))/(np.max(x)-np.min(x)),30+ysca-ysca*(y0-np.min(y))/(np.max(y)-np.min(y)),"ro",zorder=2)
        plt.xticks(size=16)
        plt.yticks(size=16)

        if sparse:
            n1 = "_sparse.jpeg"
        else:
            n1 = ".jpeg"

        plt.savefig(direc+"/PCombo_data_"+str(i)+n1,bbox_inches='tight')

#===========================================================
# Names of parameters used in plots
#===========================================================

def naming(mode):

    if mode == "Fast_Fiji":
        names = [r"$\mathrm{Base \,\, rate}\,\,(B)$",
                r"$\mathrm{Support\,\,access}\,\,(\beta_1)$",
                r"$\mathrm{Compatibility \,\,with\,\, needs}\,\, (\beta_2)$",
                r"$\mathrm{National \,\,policies}\,\,(\beta_3)$",
                r"$\mathrm{Observable \,\, benefits}\,\,(\beta_4)$",
              r"$\mathrm{Overall \,\, benefits}\,\,(\beta_5)$",
              r"$\mathrm{Presence \,\, of \,\, champions}\,\,(\beta_6)$",
              r"$\mathrm{Decision\,\, structures}\,\,(\beta_7)$",
              r"$\mathrm{Environmental \,\, conditions}\,\,(\beta_8)$",
              r"$\mathrm{Chiefly\,\, village\,\,status}\,\,(\beta_9)$",
              r"$\mathrm{Supportive \,\,institutions}\,\,(\beta_{10})$",
              r"$\mathrm{Qoliqoli\,\, Mangrove}\,\,(\beta_{11})$",
              r"$\mathrm{Qoliqoli\,\, coral}\,\,(\beta_{12})$",
              r"$\mathrm{Knowledge \,\, (coral \,\, reefs)}\,\,(\beta_{13})$",
              r"$\mathrm{Knowledge \,\, (quantity \,\,of \,\,fish)}\,\,(\beta_{14})$",
              r"$\mathrm{Knowledge \,\, (diversity\,\, of\,\, fish)}\,\,(\beta_{15})$",
              r"$\mathrm{Fairness}\,\,(\beta_{16})$",
              r"$\mathrm{Fairness}\,\,(\beta_{17})$",
              r"$\mathrm{Neighbours}\,\,(\beta_{N})$",
              r"$\mathrm{NGOs\,\,}(\beta_{M})$",
              r"$\mathrm{Qoliqolis\,\,}(\beta_{Q})$",
              r"$\mathrm{Adopting \,\, chiefly\,\,village}\,\,(\beta_{C})$",
              r"$\mathrm{Base\,\,rate}\,\,(B_{I})$",r"$\mathrm{Neighbours}\,\,(\beta_{NI}/10)$",
              r"$\mathrm{NGOs}\,\,(\beta_{MI}/10)$",
              r"$\mathrm{Qoliqolis}\,\,(\beta_{QI}/10)$",
              ]
    elif mode == "Fast_Fiji2":
        names = [r"$\mathrm{Base \,\, rate}\,\,(B)$",
                r"$\mathrm{Neighbours}\,\,(\beta_{N})$",
                r"$\mathrm{NGOs\,\,}(\beta_{M})$",
                r"$\mathrm{Qoliqolis\,\,}(\beta_{Q})$",
                r"$\mathrm{Chief}\,\,(\beta_{C})$",
                r"$\mathrm{Base\,\,rate}\,\,(B_{I})$",r"$\mathrm{Neighbours}\,\,(\beta_{NI}/10)$",
                r"$\mathrm{NGOs}\,\,(\beta_{MI}/10)$",
                r"$\mathrm{Qoliqolis}\,\,(\beta_{QI}/10)$"
              ]
    elif mode == "Fast_Fiji3":
        names = [r"$\mathrm{Base \,\, rate}\,\,(B)$",
                 r"$\mathrm{Qoliqoli\,\, Mangrove}\,\,(\beta_{11})$",
                r"$\mathrm{Qoliqoli\,\, coral}\,\,(\beta_{12})$",
                r"$\mathrm{Neighbours}\,\,(\beta_{N})$",
                r"$\mathrm{NGOs\,\,}(\beta_{M})$",
                r"$\mathrm{Qoliqolis\,\,}(\beta_{Q})$",
                r"$\mathrm{Chief}\,\,(\beta_{C})$",
                r"$\mathrm{Base\,\,rate}\,\,(B_{I})$",r"$\mathrm{Neighbours}\,\,(\beta_{NI}/10)$",
                r"$\mathrm{NGOs}\,\,(\beta_{MI}/10)$",
                r"$\mathrm{Qoliqolis}\,\,(\beta_{QI}/10)$"
                ]

    return names

#===========================================================
# Create summary for Gibbs sampler
#===========================================================

def gibbs_sum(direc,par,mode,burnin=1000,s1=1000,s2=100,s3=1000,info=0,only_measures=False,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,Future=20,only_future=False,changes=None,labels=[], sparse = False):

    filename = glob("./"+direc+"/*_00002*35_Laplace.txt")[0]

    print(filename)

    raw = np.genfromtxt(filename)[:,par:-1]

    if sparse:
        raw[abs(raw) < 1e-2] = 0

    samples = raw[burnin:,:].copy()

    names = naming(mode)

    if not only_future and not only_measures:

        plt.figure(figsize=(20,8))
        print("Making trace plots")

        for i in range(par):
            plt.subplot(4,8,i+1)
            plt.plot(samples[:,i],alpha=0.5)
            plt.xlabel("Iterations",fontsize=12)
            plt.ylabel(names[i],fontsize=12,labelpad=-2)

            plt.xticks(size=12)
            plt.yticks(size=12)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.05  )

        if sparse:
            n1 = "trace_sparse.jpeg"
        else:
            n1 = "trace.jpeg"

        plt.savefig(direc+n1, bbox_inches='tight', facecolor='w')

    if not only_future or only_measures:

        logp = np.genfromtxt(glob("./"+direc+"/*_00002*35_Laplace.txt")[0])[burnin:,-1]

        Best, IB = forest_plot(direc,raw,-logp,names,burnin,mode,sparse=sparse,only_measures=only_measures)

        best_param = np.median(samples,axis=0)

        print(best_param)

        RESd, RES2, HM, P, x, y,adag, AD,best,best_last = postpred(direc,best_param,mode,samples,s1,s2,Best,IB,info,communicate_ne,communicate_ngo,communicate_q,communicate_base,sparse=sparse,only_measures=only_measures)

#    if not only_future and not only_measures:
# 
#        if mode == "Fast_Fiji":
# 
#            mapf(direc,HM,P,x,y,adag,AD,best,best_last,sparse=sparse)

    if not only_measures:

        future_coala(direc, samples, mode, Future, s3, communicate_ne,communicate_ngo,communicate_q,communicate_base,changes,labels,info=info,sparse=sparse)

#===========================================================
# Create summary for MCMC
#===========================================================

def mcmc_sum(direc, mode, burnin=5000,s1=1000,s2=100,s3=1000,info=0,only_measures=False,communicate_ne=True,communicate_ngo=True, communicate_q=True,communicate_base=True,Future=20,changes=None,only_future=False,labels=[],media=False,truth=[]):

    samples = np.load(direc+"/samples.npy")

    ndim = np.shape(samples)[1]
    nwalkers = ndim*2
    print(ndim,nwalkers)
    print(len(samples))
    logp = np.load(direc+"/lnL.npy")

    names = naming(mode)

    # Looking for and excluding stuck walkers

    Ls = []

    for j in range(nwalkers):
        ls = logp[j::nwalkers].copy()
        ls = np.mean(ls[burnin:])
        print(ls)
        Ls.extend([ls])

    Ls = np.asarray(Ls)
    Li = []
    discard = 0

    for j, ls in enumerate(Ls):
        lij = np.arange(len(Ls))
        Lsj = np.delete(lij,j)
        if abs(ls) < abs(3*np.mean(Ls[Lsj])):
            Li.extend([j])
        else:
            print("Walker excluded: ", j,np.mean(Ls[Lsj]),ls)
            discard +=1

    # Trace plot

    burnin = burnin*nwalkers

    if not only_future and not only_measures:
        plt.figure(figsize=(20,8))
        print("Making trace plots")
        samples_nb = samples[burnin:,:].copy()
        for i in range(ndim):
            plt.subplot(4,8,i+1)
            print(names[i])
#            plt.axvline(x=burnin/nwalkers,linestyle="--",color="grey")
            for j in range(nwalkers):
                plt.plot(samples_nb[j::nwalkers,i],alpha=0.5)
            plt.xlabel("Iterations",fontsize=12)
            plt.ylabel(names[i],fontsize=12,labelpad=-2)

            plt.xticks(size=12)
            plt.yticks(size=12)
 
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.05  )

        plt.savefig(direc+"trace.jpeg", bbox_inches='tight', facecolor='w')

        print("Making pair plot")

    samples_fine = []

    for j in Li:
        dummy = samples[j::nwalkers,:]
        dummy = dummy[int(burnin/nwalkers):]
        if samples_fine == []:
            samples_fine = dummy.copy()
        else:
            samples_fine = np.vstack((samples_fine,dummy))

    if not only_future and not only_measures:

        if False:
            GTC = pygtc.plotGTC(chains=[samples_fine],   #samples[burnin:,:]],
                figureSize='MNRAS_page',
                paramNames=names,
                plotName=direc+"/mcmc.jpeg",
                legendMarker='All',
#                do1dPlots=False,
#                sigmaContourLevels=True,
                labelRotation=(0,0),
                truthColors=["#FF0000"],
                truthLineStyles=["--"],
                customTickFont={'family':'Arial', 'size':12},
                colorsOrder=["yellows","purples"],
                customLabelFont={'family':'Arial', 'size':15}
                )

    if not only_future or only_measures:

        Best, IB = forest_plot(direc,samples,logp,names,burnin,mode,samples_fine=samples_fine,only_measures=only_measures,truth=truth)

        best_param = np.median(samples_fine,axis=0)

        print("Beginning posterior predictive")

        RESd, RES2, HM, P, x, y,adag, AD,best,best_last = postpred(direc,best_param,mode,samples_fine,s1,s2,Best,IB,info,communicate_ne,communicate_ngo,communicate_q,communicate_base,only_measures=only_measures)

#    if not only_future and not only_measures:
#        if mode == "Fast_Fiji":
#   
#            mapf(direc,HM,P,x,y,adag,AD,best,best_last)

    if not only_measures:
        print(Future)

        future_coala(direc, samples_fine, mode , Future, s3, communicate_ne,communicate_ngo,communicate_q,communicate_base,changes,labels,info=info)

    if media:
        labels2 = ["Posterior predictive (sABM)", r"No media", r"Only media", r"No media in future"]
        future_media(direc, samples_fine, mode , Future, s3, communicate_ne,communicate_ngo,communicate_q,communicate_base,labels2,info=0)

#===========================================================

#changes1 = -1*np.ones(17)
#changes1[5] = 1 # index -1 because we are counting indexes of X not of beta
#changes1[7] = 1
#changes1[4] = 1
#changes2 = -1*np.ones(17)
#changes2[5] = 0
#changes2[7] = 0
#changes2[4] = 0
#changes = np.column_stack((changes1,changes2))
#labels = ["Posterior predictive (sABM)", r"More champions $(\beta_6)$",r"No champions $(\beta_6)$"] # 5
#labels = ["Posterior predictive (sABM)", r"Hotels further away $(\beta_8)$",r"Hotels closer $(\beta_8)$"] # 7
#labels = ["Posterior predictive (sABM)", r"Higher overall benefit $(\beta_5)$",r"Lower overall benefit $(\beta_5)$"] # 7

#sam = 1000

#only_measures = False
#media = True # True

#truth = np.load("truth.npy")

#mcmc_sum("./MOCK01/", "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=50000,info=0,communicate_ne=False,communicate_ngo=False,communicate_q=False, communicate_base=True,Future=28,only_future=False,changes=changes,labels=labels,only_measures=False,truth=truth)

#mcmc_sum("./MRUN10/", "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=50000,info=0,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=28,only_future=True,changes=changes,labels=labels,only_measures=only_measures,media=media)

#mcmc_sum("./MRUN10/", "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=50000,info=0,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=28,only_future=False,changes=changes,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN13/", "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=150000,info=0,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=28,only_future=False,changes=changes,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN14/", "Fast_Fiji2", s1=sam,s2=10*sam, s3 = sam, burnin=10000,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True, info = 0.0, Future=40,only_future=False,changes=None,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN11/", "Fast_Fiji2", s1=sam,s2=10*sam, s3 = sam, burnin=10000,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=40,only_future=False,changes=None,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN15/", "Fast_Fiji", s1=sam, s2=10*sam, s3=sam, burnin=50000,info=0,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=25,only_future=False,changes=changes,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN12/", "Fast_Fiji3", s1=sam,s2=10*sam, s3 = sam, burnin=20000,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=40,only_future=False,changes=None,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN16/", "Fast_Fiji2", s1=sam,s2=10*sam, s3 = sam, burnin=20000,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=40,only_future=False,changes=None,labels=labels,only_measures=only_measures)

#mcmc_sum("./MRUN17/", "Fast_Fiji3", s1=sam,s2=10*sam, s3 = sam, burnin=20000,communicate_ne=True,communicate_ngo=True,communicate_q=True, communicate_base=True,Future=40,only_future=False,changes=None,labels=labels,only_measures=only_measures)

#for sparse in [False]: #,True]:

#    gibbs_sum("./RUN30/",26,"Fast_Fiji",s1=sam,s2=10*sam,s3 =sam , burnin=50000,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,changes=changes,labels=labels,sparse=sparse,only_measures=only_measures)

#    gibbs_sum("./RUN31/",26,"Fast_Fiji",s1=sam,s2=10*sam,s3 =sam , burnin=50000,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,changes=changes,labels=labels,sparse=sparse,only_measures=only_measures)

#    gibbs_sum("./RUN32/",9,"Fast_Fiji2",s1=sam,s2=10*sam,s3 =sam , burnin=10000,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,sparse=sparse,only_measures=only_measures)

#    gibbs_sum("./RUN33/",9,"Fast_Fiji2",s1=sam,s2=10*sam,s3 =sam , burnin=18500,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,sparse=sparse,only_measures=only_measures)

#    gibbs_sum("./RUN34/",11,"Fast_Fiji3",s1=sam,s2=10*sam,s3 =sam , burnin=10000,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,sparse=sparse,only_measures=only_measures)

#    gibbs_sum("./RUN35/",11,"Fast_Fiji3",s1=sam,s2=10*sam,s3 =sam , burnin=30000,communicate_ne=True,communicate_ngo=True,communicate_q=True,communicate_base=True,sparse=sparse,only_measures=only_measures)
