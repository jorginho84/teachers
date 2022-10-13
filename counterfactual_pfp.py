#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements the pay-for-percentile (PFP) contest of Barvely and Neal (2012)
"""



from __future__ import division
import numpy as np
import pandas as pd
import pickle
import itertools
import sys, os
from scipy import stats
#from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from joblib import Parallel, delayed
from scipy import interpolate
import matplotlib.pyplot as plt
#sys.path.append("C:\\Users\\Jorge\\Dropbox\\Chicago\\Research\\Human capital and the household\]codes\\model")
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
#import gridemax
import time
#import int_linear
import between
import utility as util
import parameters_pfp as parameters
import simdata as sd
import estimate as est
from utility_counterfactual_pfp import Count_3
import simdata_c as sdc
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
from scipy import interpolate
import time
import openpyxl
from localreg import *
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")

np.random.seed(100)

#Betas and var-cov matrix
betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v23.npy")

data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')

data = data_1[data_1['d_trat']==1]

N = np.array(data['experience']).shape[0]

n_sim = 500

simce = []
baseline_p = []
income = []

#----------Percentile cutoffs--------#

data_aux =  pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pfp.dta')

experience = np.array(data_aux['experience'])

score_past = np.array(data_aux['score_past'])

n_pfp = score_past.shape[0]

nq = 10

#Experience categories
cat_exp = np.zeros(n_pfp)
cat_exp[experience <= 4] = 0
cat_exp[(experience >= 5) & (experience <= 10)] = 1
cat_exp[(experience >= 11) & (experience <= 20)] = 2
cat_exp[(experience >= 21) & (experience <= 30)] = 3
cat_exp[experience >= 40] = 4

percs_exp = np.zeros(n_pfp)


for i in range(5):
    percs_exp[cat_exp == i] = pd.qcut(score_past[cat_exp == i],nq,labels=False)
    

cutoffs_min = []
cutoffs_max = []

for j in range(5): #experience
    cutoffs_min_aux = []
    cutoffs_max_aux = []
    for i in range(nq): #percentiles

        cutoffs_min_aux.append(np.min(score_past[(cat_exp == j) & (percs_exp == i)]))
        cutoffs_max_aux.append(np.max(score_past[(cat_exp == j) & (percs_exp == i)]))
        
    cutoffs_min.append(cutoffs_min_aux)
    cutoffs_max.append(cutoffs_max_aux)


#----------Simulating counterfactual--------#

for x in range(0,2):
    

    # TREATMENT #
    treatment = np.ones(N)*x
    # EXPERIENCE #
    years = np.array(data['experience'])
    # SCORE PORTFOLIO #
    p1_0 = np.array(data['score_port_past'])
    p1 = np.array(data['score_port'])
    # SCORE TEST #
    p2_0 = np.array(data['score_port_past'])
    p2 = np.array(data['score_test'])
    # CATEGORY PORTFOLIO #
    catPort = np.array(data['cat_port'])
    # CATEGORY TEST #
    catPrueba = np.array(data['cat_test'])
    # TRAME #
    #Recover initial placement from data (2016)
    TrameI = np.array(data['trame'])
    # TYPE SCHOOL #
    typeSchool = np.array(data['typeschool'])
 
    # Priority #
    priotity = np.array(data['por_priority'])
    
    rural_rbd = np.array(data['rural_rbd'])
    
    locality = np.array(data['AsignacionZona'])
    

    #### PARAMETERS MODEL ####
    N = np.size(p1_0)
    HOURS = np.array([44]*N)
    
    alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
          betas_nelder[3], betas_nelder[4]],
         [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
          betas_nelder[8], betas_nelder[9]]]
        
    betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14]]

    gammas = [betas_nelder[15],betas_nelder[16],betas_nelder[17]]
    
    dolar= 600
    value = [14403, 15155]
    hw = [value[0]/dolar,value[1]/dolar]
    porc = [0.0338, 0.0333]
    
    #inflation adjustemtn: 2012Jan-2020Jan: 1.111
    Asig = [150000*1.111,100000*1.111,50000*1.111]
    AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 
    
    # *** This is withouth teaching career ***
    # * value professional qualification (pesos)= 72100 *
    # * value professional mention (pesos)= 24034 *
    # *** This is with teaching career ***
    # * value professional qualification (pesos)= 253076 *
    # * value professional mention (pesos)= 84360 *
    
    #inflation adjustment: 2012Jan-2019Dec: 1.266
    qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 
    qualiPesos = [72100, 24034, 253076, 84360]
    pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
    
    #* Progression component by tranche *
    #* value progression initial (pesos)= 14515
    #* value progression early (pesos)= 47831
    #* value progression advanced (pesos)= 96266
    #* value progression advanced (pesos)= 99914 Fixed component
    #* value progression expert 1 (pesos)= 360892
    #* value progression expert 1 (pesos)= 138769 Fixed component
    #* value progression expert 2 (pesos)= 776654
    #* value progression expert 2 (pesos)= 210929 Fixed component
    
    progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
    
    pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,  
       progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
    
    pri = [47872,113561]
    priori = [pri[0]/dolar, pri[1]/dolar]
    
    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori,cutoffs_min,cutoffs_max)
    
    model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality)
    
    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))
    income_sims = np.zeros((N,n_sim))
    baseline_sims = np.zeros((N,n_sim,2))
    
    for j in range(n_sim):
        modelSD = sd.SimData(N,model)
        opt = modelSD.choice()
        simce_sims[:,j] = opt['Opt Simce']
        income_sims[:,j] = opt['Opt Income'][1-x]
        baseline_sims[:,j,0] = opt['Potential scores'][0]
        baseline_sims[:,j,1] = opt['Potential scores'][1]
    
    simce.append(np.mean(simce_sims,axis=1))
    income.append(np.mean(income_sims,axis=1))
    baseline_p.append(np.mean(baseline_sims,axis=1))
    

#For validation purposes
att = simce[1] - simce[0]
att_cost = income[1] - income[0]


#Effects under a new system
treatment = np.ones(N)
    
model_c = Count_3(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                  priotity,rural_rbd,locality)
count_sd = sdc.SimDataC(N,model_c)

simce_c_sim = np.zeros((N,n_sim))
income_c_sim = np.zeros((N,n_sim))

for j in range(n_sim):
    opt = count_sd.choice()
    simce_c_sim[:,j] = opt['Opt Simce']
    income_c_sim[:,j] = opt['Opt Income'][0]

simce_c = np.mean(simce_c_sim, axis=1)
income_c = np.mean(income_c_sim, axis=1)

att_c = simce_c - simce[0]
att_cost_c = income_c - income[0]



#Effects by distance to previous test score
y = np.zeros(5)
y_c = np.zeros(5)
y_ses = np.zeros(5)
x = [1,2,3,4,5]

for j in range(5):
    y[j] = np.mean(att[data['distance2']==j+1])
    y_c[j] = np.mean(att_c[data['distance2']==j+1])
    y_ses[j] = np.std(att[data['distance2']==j+1])/att[data['distance2']==j+1].shape[0]
    



fig, ax=plt.subplots()
plot1 = ax.bar(x,y,color='b' ,alpha=.7, label = 'ATT original STPD ('+'{:04.2f}'.format(np.mean(att)) + r'$\sigma$)')
plot3 = ax.bar(x,y_c,fc= None ,alpha=.5, ec = 'darkorange',ls = '--', lw = 3,label = 'ATT modified STPD (' +'{:04.2f}'.format(np.mean(att_c)) + r'$\sigma$)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Quintiles of distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.3)
ax.legend(loc = 'upper left',fontsize = 13)
#ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1),fontsize=12,ncol=3)
plt.tight_layout()
plt.show()
fig.savefig('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/counterfactual_percentiles.pdf', format='pdf')