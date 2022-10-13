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
#sys.path.append("C:\\Users\pjdea\OneDrive\Documentos\GitRepository\wtp_table7")
#import gridemax
import time
#import int_linear
import between
import utility as util
import parameters_pfp as parameters
import simdata as sd
import estimate as est
from scipy.optimize import minimize
from utility_counterfactual import Count_1
from utility_counterfactual_exp import Count_2
from utility_counterfactual_pfp import Count_3
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
#betas_nelder  = np.load("C:\\Users\pjdea\OneDrive\Documentos\GitRepository\wtp_table7/betasopt_model_v23.npy")

data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')
#data_1 = pd.read_stata('C:\\Users\pjdea\OneDrive\Documentos\GitRepository\wtp_table7/data_pythonpast.dta')

data = data_1[data_1['d_trat']==1]

N = np.array(data['experience']).shape[0]

n_sim = 500

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

param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori,cutoffs_min,cutoffs_max)



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


#Counterfactual: pay for percentile
treatment = np.ones(N)
util_c_3 = np.zeros((N,n_sim))
income_c_3 = np.zeros((N,n_sim))
simce_c_3 = np.zeros((N,n_sim))

model_c_3 = Count_3(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                     priotity,rural_rbd,locality)



count_perc = sdc.SimDataC(N,model_c_3)             
for j in range(n_sim):
    opt = count_perc.choice()
    util_c_3[:,j] = opt['Opt Utility']
    income_c_3[:,j] = opt['Opt Income'][0]
    simce_c_3[:,j] = opt['Opt Simce']

    

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
    
    #wtp w/r to linear Pay for percentile
    wtp_list.append(np.exp(util_c_3[:,j] - (gammas[0]*effort_m + gammas[1]*effort_h + gammas[2]*np.log(opt['Opt Student H']))) - opt['Opt Income'][0])
    

    #Changes in income (to compute added revenues)
    income_list.append(income_stpd[:,j] - income_0[:,j])
    income_list.append(income_c_2[:,j] - income_0[:,j])
    income_list.append(income_c_1[:,j] - income_0[:,j])
    income_list.append(income_c_3[:,j] - income_0[:,j])
    
    #ATTs on SIMCE
    simce_list.append(simce_stpd[:,j] - simce_0[:,j])
    simce_list.append(simce_c_2[:,j] - simce_0[:,j])
    simce_list.append(simce_c_1[:,j] - simce_0[:,j])
    simce_list.append(simce_c_3[:,j] - simce_0[:,j])
                                 

#Average WTPs: saving them into table

#WTPs
original_stpd_O7 = np.mean(wtp_list[0])
no_experience_P7 = np.mean(wtp_list[1])
linear_Q7 = np.mean(wtp_list[2])
percentile_Q7 = np.mean(wtp_list[3])

#Delta income (also provision cost)
original_stpd_S7 = np.mean(income_list[0])
no_experience_T7 = np.mean(income_list[1])
linear_U7 = np.mean(income_list[2])
percentile_U7 = np.mean(income_list[3])

#ATTs
STPD_C22 = np.mean(simce_list[0])
no_experience_D22 = np.mean(simce_list[1])
linear_pfp_E22 = np.mean(simce_list[2])
percentile_E22 = np.mean(simce_list[3])

ATT_E4 = 0.22
first_gain_F4 = 0.1
marginal_tax_rate_D4 = 0.35


#Averge annual wage (2020 dollars, jan2020-jan2002)
av_annual_wage = 4565*(28310.86/16262.66)

wage = np.zeros(40)
interes = 0.03

for i in range(40):
    wage[i] = av_annual_wage/((1+interes)**(i))
    

lifetime_earnings_G4 = np.sum(wage)


#taken from bravo, hojman, and rodriguez
#lifetime_earnings_G4 = 108684.5

#STUDENTS 
studentst7 = np.zeros(4)
v_students = np.array([STPD_C22, no_experience_D22, linear_pfp_E22,percentile_E22])

for i in range(4):
    studentst7[i] = v_students[i]*first_gain_F4*lifetime_earnings_G4*(1-marginal_tax_rate_D4)
    
#TEACHERS 
teacherst7 = np.zeros(4)
v_teachers = np.array([original_stpd_O7, no_experience_P7, linear_Q7,percentile_Q7])

for i in range(4):
    teacherst7[i] = v_teachers[i]*12*(1-marginal_tax_rate_D4)

#OVERAL WTP 
overalwtpt7 = np.zeros(4)

for i in range(4):
    overalwtpt7[i] = studentst7[i] + teacherst7[i]
    
# PROVISION COST 
provisiont7 = np.zeros(4)
v_provision = np.array([original_stpd_S7, no_experience_T7, linear_U7,percentile_U7])

for i in range(4):
    provisiont7[i] = v_provision[i]*12
    
# ADDED REVENUE
revenuet7 = np.zeros(4)

for i in range(4):
    revenuet7[i] = v_students[i]*first_gain_F4*lifetime_earnings_G4*marginal_tax_rate_D4 + provisiont7[i]*marginal_tax_rate_D4

# NET COST
net_costt7 = np.zeros(4)

for i in range(4):
    net_costt7[i] = provisiont7[i] - revenuet7[i]
    
# MVPF
mvpft7 = np.zeros(4)

for i in range(4):
    mvpft7[i] = overalwtpt7[i]/net_costt7[i]


#\Latex archive
#Table 7 of the paper

with open('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/wtp_table7.tex','w') as f:
    f.write(r'\footnotesize{'+'\n')
    f.write(r'\begin{tabular}{lcccccccc}'+'\n')
    f.write(r'\toprule'+'\n')
    f.write(r'&  & \multirow{2}{*}{\makecell[c]{\textbf{Original} \\ \textbf{STPD}}} & & \multirow{2}{*}{\makecell[c]{\textbf{Policy 1} \\ \textbf{(no experience)}}} & & \multirow{2}{*}{\makecell[c]{\textbf{Policy 2} \\ \textbf{(linear PFP)}}} & & \multirow{2}{*}{\makecell[c]{\textbf{Policy 3} \\ \textbf{(percentiles)}}} \\'+'\n')
    f.write(r'& &  & &  & & & & \\'+'\n')
    f.write(r'\midrule'+'\n')
    f.write(r'\textbf{A. Willigness to pay}  & &  & &  & & & &\\'+'\n')
    f.write(r'\multirow{2}{*}{\makecell[l]{Students WTP (in \$): \\ $ATT\times \rho \times \$258,272\times$  $(1-\tau)$}} &  & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(studentst7[0]) +r'}} & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(studentst7[1]) +r'}} & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(studentst7[2]) +r'}}  & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(studentst7[3]) +r'}}\\'+'\n')
    f.write(r'& &  & &  & &   & & \\'+'\n')
    f.write(r'& &  & &  & &   & & \\'+'\n')
    f.write(r'Teachers WTP (in \$)  & & '+'{:1.0f}'.format(teacherst7[0]) +r' & & '+'{:1.0f}'.format(teacherst7[1]) +r' & & '+'{:1.0f}'.format(teacherst7[2]) +r' & & '+'{:1.0f}'.format(teacherst7[3]) +r'\\'+'\n')
    f.write(r'& &  & &  & & & &\\'+'\n')
    f.write(r'Overall WTP  & & '+'{:1.0f}'.format(overalwtpt7[0]) +r' & & '+'{:1.0f}'.format(overalwtpt7[1]) +r' & & '+'{:1.0f}'.format(overalwtpt7[2]) +r' & & '+'{:1.0f}'.format(overalwtpt7[3])+r'\\'+'\n')
    f.write(r'& &  & &  & &  & & \\'+'\n')
    f.write(r'& &  & &  & & & & \\'+'\n')
    f.write(r'\textbf{B. Costs} & &  & &  & & & & \\'+'\n')
    f.write(r'Provision cost $C$ (in \$) & & '+'{:1.0f}'.format(provisiont7[0]) +r' & & '+'{:1.0f}'.format(provisiont7[1]) +r' & & '+'{:1.0f}'.format(provisiont7[2]) +r' & & '+'{:1.0f}'.format(provisiont7[3])+r'\\'+'\n')
    f.write(r'& &  & &  & & & &\\'+'\n')
    f.write(r'\multirow{2}{*}{\makecell[l]{Added revenues (in \$)} &  & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(revenuet7[0]) +r'}} & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(revenuet7[1]) +r'}} & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(revenuet7[2]) +r'}} & & \multirow{2}{*}{\makecell[l]{'+'{:1.0f}'.format(revenuet7[3]) +r'}} \\'+'\n')
    f.write(r'& &  & &  & & & & \\'+'\n')
    f.write(r'& &  & &  & & & & \\'+'\n')
    f.write(r'Net Cost & & '+'{:1.0f}'.format(net_costt7[0]) +r' & & '+'{:1.0f}'.format(net_costt7[1]) +r' & & '+'{:1.0f}'.format(net_costt7[2]) +r' & & '+'{:1.0f}'.format(net_costt7[3])  +r' \\'+'\n')
    f.write(r'\midrule'+'\n')
    f.write(r'\textbf{MVPF: WTP/Net Cost} & & \textbf{'+'{:1.2f}'.format(mvpft7[0]) +r'} & & \textbf{'+'{:1.2f}'.format(mvpft7[1]) +r'} & & \textbf{'+'{:1.2f}'.format(mvpft7[2]) +r'} & & \textbf{'+'{:1.2f}'.format(mvpft7[3]) +r'} \\ '+'\n')
    f.write(r'\bottomrule'+'\n')
    f.write(r'\end{tabular}'+'\n')
    f.write(r'}'+'\n')
    f.close()
 



