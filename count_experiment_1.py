# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:06 2021

This .py generates counterfactual experiment #1: a policy that eliminates progression in the system

exec(open("/home/jrodriguezo/teachers/codes/count_experiment_1.py").read())

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
#sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
sys.path.append("/home/jrodriguezo/teachers/codes")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
from utility_counterfactual_att import Count_att_2
from utility_counterfactual_att_categories import Count_att_2_cat
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
from scipy import interpolate
import time
import openpyxl

# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
import linearmodels as lm
from linearmodels.panel import PanelOLS


np.random.seed(123)


#----------------------------------------------#
#----------------------------------------------#
#Obtaining simulated effects
#
#----------------------------------------------#
#----------------------------------------------#

#betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/betasopt_model_v40.npy")
betas_nelder  = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v47.npy")

#Only treated teachers
#data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data_1 = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')
data = data_1[data_1['d_trat']==1]
N = np.array(data['experience']).shape[0]

n_sim = 500


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
TrameI = np.array(data['tramo_a2016'])
# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

# Priority #
priotity = np.array(data['por_priority'])

priotity_aep = np.array(data['priority_aep'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])


#### PARAMETERS MODEL ####
N = np.size(p1_0)
HOURS = np.array([44]*N)

alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
         betas_nelder[3], betas_nelder[4]],
        [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
        betas_nelder[8], betas_nelder[9]]]
        
betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12],betas_nelder[13],betas_nelder[14],betas_nelder[15]]
gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]


dolar= 600
value = [14403, 15155]
hw = [value[0]/dolar,value[1]/dolar]
porc = [0.0338, 0.0333]

#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [50000*1.111, 100000*1.111, 150000*1.111]
AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 

# *** This is withouth teaching career ***
# * value professional qualification (pesos)= 72100 *
# * value professional mention (pesos)= 24034 *
# *** This is with teaching career ***
# * value professional qualification (pesos)= 253076 *
# * value professional mention (pesos)= 84360 *

#inflation adjustemtn: 2012Jan-2020Jan: 1.111***
qualiPesos = [72100*1.111, 24034*1.111, 253076, 84360] 
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

pri = [48542,66609,115151]
priori = [pri[0]/dolar, pri[1]/dolar, pri[2]/dolar]

param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)

#List of choices across counterfactuals
simce = []
baseline_p = []
effort_p = []
effort_t = []


#Original ATT
for x in range(0,3):

	


    
    # TREATMENT #

    if x <= 1:
    	treatment = np.ones(N)*x
    	 #Original STPD
    	model = Count_att_2(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality, priotity_aep)
    else:
    	treatment = np.ones(N)
    	model = Count_att_2_cat(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality, priotity_aep)

  
    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))
    baseline_sims = np.zeros((N,n_sim,2))
    effort_p_sims = np.zeros((N,n_sim))
    effort_t_sims = np.zeros((N,n_sim)) 
    
    for j in range(n_sim):
        modelSD = sd.SimData(N,model)
        opt = modelSD.choice()
        simce_sims[:,j] = opt['Opt Simce']
        effort_v1 = opt['Opt Effort']
        d_effort_t1 = effort_v1 == 1
        d_effort_t2 = effort_v1 == 2
        d_effort_t3 = effort_v1 == 3
        
        effort_m = d_effort_t1 + d_effort_t3
        effort_h = d_effort_t2 + d_effort_t3
        effort_p_sims[:,j] = effort_m
        effort_t_sims[:,j] = effort_h
    
    simce.append(np.mean(simce_sims,axis=1))
    baseline_p.append(np.mean(baseline_sims,axis=1))
    effort_p.append(np.mean(effort_p_sims,axis = 1))
    effort_t.append(np.mean(effort_t_sims,axis = 1))    



print ('')
print ('ATT equals ', np.mean(simce[1] - simce[0]))
print ('')


#For validation purposes
att_sim_original = simce[1] - simce[0]
att_sim_count = simce[2] - simce[0]

#Initial categorization
initial_p = np.zeros(N)
initial_p[(TrameI=='INICIAL')] = 1
initial_p[(TrameI=='TEMPRANO')] = 2
initial_p[(TrameI=='AVANZADO')] = 3
initial_p[(TrameI=='EXPERTO I')] = 4
initial_p[(TrameI=='EXPERTO II')] = 5


#---------------------------------------------------------------#
#Effects by initial categorization
#---------------------------------------------------------------#

y = np.zeros(5)
y_c = np.zeros(5)
x = [1,2,3,4,5]

for j in range(5):
	y[j] = np.mean(att_sim_original[initial_p == j + 1])
	y_c[j] = np.mean(att_sim_count[initial_p == j + 1])

fig, ax=plt.subplots()
plot1 = ax.bar(x,y,color='b' ,alpha=.9, label = 'ATT original STPD ('+'{:04.2f}'.format(np.mean(att_sim_original)) + r'$\sigma$)')
plot3 = ax.bar(x,y_c,fc= None ,alpha=.6, lw = 3,label = 'ATT modified STPD (' +'{:04.2f}'.format(np.mean(att_sim_count)) + r'$\sigma$)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Initial categorization', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(x, ['Initial', 'Early', 'Advanced', 'Expert I', 'Expert II'],fontsize=12)
#ax.set_ylim(0,0.26)
ax.legend(loc = 'upper left',fontsize = 13)
#ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1),fontsize=12,ncol=3)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/att_count_categories.pdf', format='pdf')



