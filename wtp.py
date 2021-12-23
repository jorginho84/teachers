# -*- coding: utf-8 -*-
"""
This code computes WTPs for different policies
"""


from __future__ import division
import numpy as np
import pandas as pd
import pickle
import itertools
import sys, os
from scipy import stats
#from scipy.optimize import minimize
import scipy.optimize
from scipy.optimize import newton_krylov
from scipy.optimize import fmin_bfgs
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize_scalar
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
import parameters as parameters
import simdata as sd
import estimate as est
from scipy.optimize import minimize
from utility_counterfactual import Count_1
from utility_counterfactual_exp import Count_2
import simdata_c as sdc
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
from scipy import interpolate
import time



np.random.seed(100)

#Betas and var-cov matrix
betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v23.npy")

data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')

data = data_1[data_1['d_trat']==1]

N = np.array(data['experience']).shape[0]

n_sim = 500



#Baseline policy WTP#

utility_policies = []
income_policies = []



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

param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)



#STPD original
treatment = np.ones(N)
util_stpd = np.zeros((N,n_sim))
income_stpd = np.zeros((N,n_sim))
simce_stpd = np.zeros((N,n_sim))

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                     priotity,rural_rbd,locality)


modelSD = sd.SimData(N,model)
for j in range(n_sim):
    opt = modelSD.choice()
    util_stpd[:,j] = opt['Opt Utility']
    income_stpd[:,j] = opt['Opt Income'][0]
    simce_stpd[:,j] = opt['Opt Simce']
    

#Counterfactual: experience
treatment = np.ones(N)
util_c_2 = np.zeros((N,n_sim))
income_c_2 = np.zeros((N,n_sim))
simce_c_2 = np.zeros((N,n_sim))

model_c_2 = Count_2(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                     priotity,rural_rbd,locality)

count_exp = sdc.SimDataC(N,model_c_2)
               
for j in range(n_sim):
    opt = count_exp.choice()
    util_c_2[:,j] = opt['Opt Utility']
    income_c_2[:,j] = opt['Opt Income'][0]
    simce_c_2[:,j] = opt['Opt Simce']


#Counterfactual: linear PFP
treatment = np.ones(N)
util_c_1 = np.zeros((N,n_sim))
income_c_1 = np.zeros((N,n_sim))
simce_c_1 = np.zeros((N,n_sim))

model_c_1 = Count_1(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                     priotity,rural_rbd,locality)



count_pfp = sdc.SimDataC(N,model_c_1)             
for j in range(n_sim):
    opt = count_pfp.choice()
    util_c_1[:,j] = opt['Opt Utility']
    income_c_1[:,j] = opt['Opt Income'][0]
    simce_c_1[:,j] = opt['Opt Simce']



#Pre-reform and WTP calculations#

#recovering optimal pre-reform choices
treatment = np.zeros(N)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                     priotity,rural_rbd,locality)

wtp = np.zeros((N,n_sim))
util_0 = np.zeros((N,n_sim))
income_0 = np.zeros((N,n_sim))
simce_0 = np.zeros((N,n_sim))

wtp_list = []
income_list = []
simce_list = []

for j in range(n_sim):
    modelSD = sd.SimData(N,model)
    opt = modelSD.choice()
    util_0[:,j] = opt['Opt Utility']
    income_0[:,j] = opt['Opt Income'][1] #income under treatment = 0
    simce_0[:,j] = opt['Opt Simce']
    
   
    d_effort_t1 = opt['Opt Effort'] == 1
    d_effort_t2 = opt['Opt Effort'] == 2
    d_effort_t3 = opt['Opt Effort'] == 3
    effort_m = d_effort_t1 + d_effort_t3
    effort_h = d_effort_t2 + d_effort_t3
    
    #wtp w/r to STPD
    wtp_list.append(np.exp(util_stpd[:,j] - (gammas[0]*effort_m + gammas[1]*effort_h + gammas[2]*np.log(opt['Opt Student H']))) - opt['Opt Income'][0])
    
    #wtp w/r to STPD w/o experience
    wtp_list.append(np.exp(util_c_2[:,j] - (gammas[0]*effort_m + gammas[1]*effort_h + gammas[2]*np.log(opt['Opt Student H']))) - opt['Opt Income'][0])
    
     #wtp w/r to linear PFP
    wtp_list.append(np.exp(util_c_1[:,j] - (gammas[0]*effort_m + gammas[1]*effort_h + gammas[2]*np.log(opt['Opt Student H']))) - opt['Opt Income'][0])

    #Changes in income (to compute added revenues)
    income_list.append(income_stpd[:,j] - income_0[:,j])
    income_list.append(income_c_2[:,j] - income_0[:,j])
    income_list.append(income_c_1[:,j] - income_0[:,j])
    
    #ATTs on SIMCE
    simce_list.append(simce_stpd[:,j] - simce_0[:,j])
    simce_list.append(simce_c_2[:,j] - simce_0[:,j])
    simce_list.append(simce_c_1[:,j] - simce_0[:,j])
                                 

#Average WTPs: saving them into table
np.mean(wtp_list[0])

wb = load_workbook('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/mvpf_teachers.xlsx')
sheet = wb["calculos"]

sheet['O7'] = np.mean(wtp_list[0])
sheet['P7'] = np.mean(wtp_list[1])
sheet['Q7'] = np.mean(wtp_list[2])

sheet['S7'] = np.mean(income_list[0])
sheet['T7'] = np.mean(income_list[1])
sheet['U7'] = np.mean(income_list[2])

sheet['C22'] = np.mean(simce_list[0])
sheet['D22'] = np.mean(simce_list[1])
sheet['E22'] = np.mean(simce_list[2])  

wb.save('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/mvpf_teachers.xlsx')


