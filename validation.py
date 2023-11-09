# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:06 2021

This .py generate two figures comparing simulated and data-based att across initial placement and distance to the performance cutoff

exec(open("/home/jrodriguezo/teachers/codes/validation.py").read())

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
from utility_counterfactual_att_2 import Count_att_2
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

#Betas and var-cov matrix

#betas_nelder = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/betasopt_model_v23.npy")
#df = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/data_main_regmain_v2023.dta')
#moments_vector = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/moments.npy")
#ses_opt = np.load('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/ses_model.npy')
#data = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/data_pythonpast_v2023.dta')

#----------------------------------------------#
#----------------------------------------------#
#Preparing data for regressions
#
#----------------------------------------------#
#----------------------------------------------#
#data_reg = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/FINALdata.dta')
data_reg = pd.read_stata('/home/jrodriguezo/teachers/data/FINALdata.dta')



# first drop Stata 1083190 rows 
data_reg = data_reg[(data_reg["stdsimce_m"].notna()) & (data_reg["stdsimce_l"].notna())]

#destring
data_reg["drun_l"] = pd.to_numeric(data_reg["drun_l"], errors='coerce')
data_reg["drun_m"] = pd.to_numeric(data_reg["drun_m"], errors='coerce')


##### generates variables #####
#eval_year
data_reg.loc[data_reg["eval_year_m"]==data_reg["eval_year_l"],'eval_year'] = data_reg["eval_year_m"]
data_reg.loc[(data_reg["eval_year_m"].notna()) & (data_reg["eval_year_l"].isna()),'eval_year'] = data_reg["eval_year_m"]
data_reg.loc[(data_reg["eval_year_m"].isna()) & (data_reg["eval_year_l"].notna()),'eval_year'] = data_reg["eval_year_l"]

#drun
data_reg.loc[data_reg["drun_m"]==data_reg["drun_l"],'drun'] = data_reg["drun_m"]
data_reg.loc[(data_reg["drun_m"].notna()) & (data_reg["drun_l"].isna()),'drun'] = data_reg["drun_m"]
data_reg.loc[(data_reg["drun_m"].isna()) & (data_reg["drun_l"].notna()),'drun'] = data_reg["drun_l"]

#experience
data_reg.loc[data_reg["experience_m"]==data_reg["experience_l"],'experience'] = data_reg["experience_m"]
data_reg.loc[(data_reg["experience_m"].notna()) & (data_reg["experience_l"].isna()),'experience'] = data_reg["experience_m"]
data_reg.loc[(data_reg["experience_m"].isna()) & (data_reg["experience_l"].notna()),'experience'] = data_reg["experience_l"]

#d_trat
data_reg.loc[data_reg["d_trat_m"]==data_reg["d_trat_l"],'d_trat'] = data_reg["d_trat_m"]
data_reg.loc[(data_reg["d_trat_m"].notna()) & (data_reg["d_trat_l"].isna()),'d_trat'] = data_reg["d_trat_m"]
data_reg.loc[(data_reg["d_trat_m"].isna()) & (data_reg["d_trat_l"].notna()),'d_trat'] = data_reg["d_trat_l"]

#inter
data_reg.loc[data_reg["inter_m"]==data_reg["inter_l"],'inter'] = data_reg["inter_m"]
data_reg.loc[(data_reg["inter_m"].notna()) & (data_reg["inter_l"].isna()),'inter'] = data_reg["inter_m"]
data_reg.loc[(data_reg["inter_m"].isna()) & (data_reg["inter_l"].notna()),'inter'] = data_reg["inter_l"]

#d_year
data_reg.loc[data_reg["d_year_m"]==data_reg["d_year_l"],'d_year'] = data_reg["d_year_m"]
data_reg.loc[(data_reg["d_year_m"].notna()) & (data_reg["d_year_l"].isna()),'d_year'] = data_reg["d_year_m"]
data_reg.loc[(data_reg["d_year_m"].isna()) & (data_reg["d_year_l"].notna()),'d_year'] = data_reg["d_year_l"]

#Distance to cutofff
data_reg.loc[data_reg["XY_distance_m"]==data_reg["XY_distance_l"],'XY_distance'] = data_reg["XY_distance_m"]
data_reg.loc[(data_reg["XY_distance_m"].notna()) & (data_reg["XY_distance_l"].isna()),'XY_distance'] = data_reg["XY_distance_m"]
data_reg.loc[(data_reg["XY_distance_m"].isna()) & (data_reg["XY_distance_l"].notna()),'XY_distance'] = data_reg["XY_distance_l"]


##### drop nan #####
data_reg = data_reg[(data_reg["edp"].notna())]
data_reg = data_reg[(data_reg["edm"].notna())]
data_reg = data_reg[(data_reg["ingreso"].notna())]
data_reg = data_reg[(data_reg["experience"].notna())]
data_reg = data_reg[(data_reg["drun"].notna())]
data_reg = data_reg[(data_reg["d_trat"].notna())]
data_reg = data_reg[(data_reg["d_year"].notna())]
data_reg = data_reg[(data_reg["inter"].notna())]
                                                                   
# keep if eval_year == 1 | eval_year == 2018 | eval_year == 0
data_reg = data_reg[(data_reg["eval_year"] == 1) | (data_reg["eval_year"] == 2018) | (data_reg["eval_year"] == 0)]

# mean simce
data_reg['stdsimce'] = data_reg[['stdsimce_m', 'stdsimce_l']].mean(axis=1)
data_reg['constant'] = np.ones(np.size(data_reg['stdsimce']))


#Distance categories
data_reg.loc[data_reg['XY_distance']<= 0.1,'distance2'] = 1
data_reg.loc[(data_reg['XY_distance']> 0.1) & (data_reg['XY_distance'] <= 0.2),'distance2'] = 2
data_reg.loc[(data_reg['XY_distance']> 0.2) & (data_reg['XY_distance'] <= 0.3),'distance2'] = 3
data_reg.loc[(data_reg['XY_distance']> 0.3) & (data_reg['XY_distance'] <= 0.4),'distance2'] = 4
data_reg.loc[(data_reg['XY_distance']> 0.4),'distance2'] = 5




#----------------------------------------------#
#----------------------------------------------#
#Obtaining simulated effects
#
#----------------------------------------------#
#----------------------------------------------#

#betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/betasopt_model_v40.npy")
betas_nelder  = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v41.npy")

#Only treated teachers
#data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data_1 = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')
data = data_1[data_1['d_trat']==1]
N = np.array(data['experience']).shape[0]

n_sim = 500

simce = []
baseline_p = []
income = []
effort_p = []
effort_t = [] 


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
    
    alphas_control = [[betas_nelder[19],betas_nelder[20]],[betas_nelder[21],betas_nelder[22]]]
    betas_control = [betas_nelder[23],betas_nelder[24]]
    
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
    
    param0 = parameters.Parameters(alphas,betas,gammas,alphas_control, betas_control,
                                   hw,porc,pro,pol,AEP,priori)
    
    model = Count_att_2(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality, priotity_aep)
    
    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))
    income_sims = np.zeros((N,n_sim))
    baseline_sims = np.zeros((N,n_sim,2))
    effort_p_sims = np.zeros((N,n_sim))
    effort_t_sims = np.zeros((N,n_sim)) 
    
    for j in range(n_sim):
        modelSD = sd.SimData(N,model)
        opt = modelSD.choice()
        simce_sims[:,j] = opt['Opt Simce'][1-x]
        income_sims[:,j] = opt['Opt Income'][1-x]
        effort_v1 = opt['Opt Effort']
        d_effort_t1 = effort_v1 == 1
        d_effort_t2 = effort_v1 == 2
        d_effort_t3 = effort_v1 == 3
        
        effort_m = d_effort_t1 + d_effort_t3
        effort_h = d_effort_t2 + d_effort_t3
        effort_p_sims[:,j] = effort_m
        effort_t_sims[:,j] = effort_h
    
    simce.append(np.mean(simce_sims,axis=1))
    income.append(np.mean(income_sims,axis=1))
    baseline_p.append(np.mean(baseline_sims,axis=1))
    effort_p.append(np.mean(effort_p_sims,axis = 1))
    effort_t.append(np.mean(effort_t_sims,axis = 1))    



print ('')
print ('ATT equals ', np.mean(simce[1] - simce[0]))
print ('')


#For validation purposes
att_sim = simce[1] - simce[0]
att_cost = income[1] - income[0]
effort_att_p = effort_p[1] - effort_p[0]
effort_att_t = effort_t[1] - effort_t[0]
att_mean_sim = np.mean(att_sim)


#Initial categorization
initial_p = np.zeros(N)
initial_p[(TrameI=='INICIAL')] = 1
initial_p[(TrameI=='TEMPRANO')] = 2
initial_p[(TrameI=='AVANZADO')] = 3
initial_p[(TrameI=='EXPERTO I')] = 4
initial_p[(TrameI=='EXPERTO II')] = 5

print ('')
print ('ATT early', np.mean(att_sim[initial_p == 2]))
print ('')


print ('')
print ('ATT advanced', np.mean(att_sim[initial_p == 3]))
print ('')


print ('')
print ('ATT expert', np.mean(att_sim[initial_p == 4]))
print ('')


#----------------------------------------------#
#----------------------------------------------#
#Effects by distance to nearest cutoff
#
#----------------------------------------------#
#----------------------------------------------#

"""

#the fact that we are comparing STPD against the previous system versus no incentives does not change the fact that we have a U shape.
#try with including previous test scores?

Y = att_sim
data['XY_distance_2'] = data['XY_distance']**2
X = data[['XY_distance','XY_distance_2']]
X = sm.add_constant(X)
model = sm.OLS(Y,X,missing = 'drop')
results = model.fit()
print(results.summary())
y_sim = results.params[0] + results.params[1]*data['XY_distance'] + results.params[2]*data['XY_distance_2']
x_points = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])
y_sim_2 = results.params[0] + results.params[1]*x_points + results.params[2]*x_points**2


fig, ax=plt.subplots()
plot3 = ax.plot(x_points,y_sim_2,'--o',alpha = .8, color='sandybrown')



data['XY_distance_3'] = data['XY_distance']**3
X = data[['XY_distance','XY_distance_2','XY_distance_3']]
X = sm.add_constant(X)
model = sm.OLS(Y,X,missing = 'drop')
results = model.fit()
print(results.summary())
y_sim = results.params[0] + results.params[1]*data['XY_distance'] + results.params[2]*data['XY_distance_2'] + results.params[3]*data['XY_distance_3']


X1 = data['XY_distance']
X1 = sm.add_constant(X1)
model = sm.OLS(Y,X1,missing = 'drop')
results = model.fit()
print(results.summary())


fig, ax=plt.subplots()
plot3 = ax.plot(data['XY_distance'],y_sim,'--o',alpha = .8, color='sandybrown')


"""



y = np.zeros(5)
y_cat = [np.zeros(5),np.zeros(5),np.zeros(5)]
simce_cat = [np.zeros(5),np.zeros(5),np.zeros(5)]
y_c = np.zeros(5)
y_ses = np.zeros(5)
x = [1,2,3,4,5]

for j in range(5):
    y[j] = np.mean(att_sim[data['distance2']== j + 1])
    y_cat[0][j] = np.mean(att_sim[(data['distance2']== j + 1) & (initial_p == 2)])
    y_cat[1][j] = np.mean(att_sim[(data['distance2']== j + 1) & (initial_p == 3)])
    y_cat[2][j] = np.mean(att_sim[(data['distance2']== j + 1) & (initial_p == 4)])
    

inter_data = np.zeros(5)
se_data = np.zeros(5)

data_reg.set_index(['rbd', 'agno'],inplace = True)
model_reg = PanelOLS(dependent = data_reg['stdsimce'], exog = data_reg[['constant', 'd_trat', 'd_year', 'inter', 'edp', 'edm', 'ingreso', 'experience']], entity_effects = True)
results = model_reg.fit(cov_type='clustered', clusters=data_reg['drun'])
att_data = results.params.inter.round(8)




data_reg.reset_index(inplace = True)
for j in range(5):
    data_reg_j = data_reg[(data_reg['distance2'] == j + 1)]
    data_reg_j.set_index(['rbd', 'agno'],inplace = True)
    model_reg = PanelOLS(dependent = data_reg_j['stdsimce'], 
                         exog = data_reg_j[['constant', 'd_trat', 'd_year', 'inter','edp', 'edm', 'ingreso', 'experience']], entity_effects = True)
    results = model_reg.fit(cov_type='clustered', clusters=data_reg_j['drun'])
    inter_data[j] = results.params.inter.round(8)
    se_data[j] = np.sqrt(results.cov.inter.inter)
    
    

#Quadratic fit from model
Y = att_sim
data['XY_distance_2'] = data['XY_distance']**2
X = data[['XY_distance','XY_distance_2']]
X = sm.add_constant(X)
model = sm.OLS(Y,X,missing = 'drop')
results = model.fit()
print(results.summary())
y_sim = results.params[0] + results.params[1]*data['XY_distance'] + results.params[2]*data['XY_distance_2']
x_points = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
y_sim_2 = results.params[0] + results.params[1]*x_points + results.params[2]*x_points**2


#Quadratic fit from data
data_reg.reset_index(inplace = True)
data_reg.set_index(['rbd', 'agno'],inplace = True)
data_reg['XY_distance_2'] = data_reg['XY_distance']**2 
data_reg['inter_xy'] = data_reg['inter']*data_reg['XY_distance']
data_reg['inter_xy_2'] = data_reg['inter']*data_reg['XY_distance_2']
model_reg = PanelOLS(dependent = data_reg['stdsimce'], 
                     exog = data_reg[['constant', 'd_trat', 'd_year', 'inter', 'inter_xy','inter_xy_2','edp', 'edm', 'ingreso', 'experience']], entity_effects = True)
results = model_reg.fit(cov_type='clustered', clusters=data_reg['drun'])
y_sim_data = results.params.inter + results.params.inter_xy*x_points + results.params.inter_xy_2*x_points**2

#a = results.params.inter.round(8)

#ATT data vs ATT model by categories
x_points_data = np.array([0.1,0.25,0.35,0.5,1])
fig, ax=plt.subplots()
plot1 = ax.plot(x_points_data,inter_data,'bo',alpha=.7,label = 'ATT data')
plot2 = ax.errorbar(x_points_data, inter_data, yerr=se_data, fmt='none', color='k')
plot3 = ax.plot(x_points_data+0.01,y_cat[0],'o',alpha = .8, color='sandybrown', label = 'ATT model (early)')
plot4 = ax.plot(x_points_data+0.02,y_cat[1],'s',alpha = .8, color='red', label = 'ATT model (advanced)')
plot5 = ax.plot(x_points_data+0.03,y_cat[2],'v',alpha = .6, color='black', label = 'ATT model (expert)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
#ax.legend(fontsize = 13)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6),fontsize=12,ncol=2)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/att_data_categories.pdf', format='pdf')


#ATT data vs ATT overall
x_points_data = np.array([0.1,0.25,0.35,0.5,1])
fig, ax=plt.subplots()
plot1 = ax.plot(x_points_data,inter_data,'bo',alpha=.7,label = 'ATT data')
plot2 = ax.errorbar(x_points_data, inter_data, yerr=se_data, fmt='none', color='k')
plot3 = ax.plot(x_points_data+0.01,y,'o',alpha = .8, color='sandybrown', label = 'ATT model')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
#ax.legend(fontsize = 13)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6),fontsize=12,ncol=2)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/att_data_model_points.pdf', format='pdf')

#ATT data vs ATT model: comparing curvature
fig, ax=plt.subplots()
plot1 = ax.plot(x_points,y_sim_data,'--b',alpha=.7,label = 'ATT data')
plot2 = ax.plot(x_points,y_sim_2,'--o',alpha = .8, color='sandybrown', label = 'ATT model')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
#ax.legend(fontsize = 13)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6),fontsize=12,ncol=2)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/att_data_model_quadratic.pdf', format='pdf')


#SIMCE (1,0)

for j in range(5):
    simce_cat[0][j] = np.mean(simce[0][(data['distance2']== j + 1) & (initial_p == 2)])
    simce_cat[1][j] = np.mean(simce[0][(data['distance2']== j + 1) & (initial_p == 3)])
    simce_cat[2][j] = np.mean(simce[0][(data['distance2']== j + 1) & (initial_p == 4)])



fig, ax=plt.subplots()
plot3 = ax.plot(x_points_data+0.01,simce_cat[0],'o',alpha = .8, color='sandybrown', label = 'ATT model (early)')
plot4 = ax.plot(x_points_data+0.02,simce_cat[1],'s',alpha = .8, color='red', label = 'ATT model (advanced)')
plot5 = ax.plot(x_points_data+0.03,simce_cat[2],'v',alpha = .6, color='black', label = 'ATT model (expert)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
#ax.legend(fontsize = 13)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6),fontsize=12,ncol=2)
plt.tight_layout()
plt.show()


#Effort (1,0)
effort_cat = [np.zeros(5),np.zeros(5),np.zeros(5)]
effort = np.zeros(5) 

for j in range(5):
    effort[j] = np.mean(effort_att_p[data['distance2']== j + 1])
    effort_cat[0][j] = np.mean(effort_att_p[(data['distance2']== j + 1) & (initial_p == 2)])
    effort_cat[1][j] = np.mean(effort_att_p[(data['distance2']== j + 1) & (initial_p == 3)])
    effort_cat[2][j] = np.mean(effort_att_p[(data['distance2']== j + 1) & (initial_p == 4)])



fig, ax=plt.subplots()
plot2 = ax.plot(x_points_data+0.01,effort*100,'o',alpha = .8, color='blue', label = 'ATT model all')
plot3 = ax.plot(x_points_data+0.01,effort_cat[0]*100,'o',alpha = .8, color='sandybrown', label = 'ATT model (early)')
plot4 = ax.plot(x_points_data+0.02,effort_cat[1]*100,'s',alpha = .8, color='red', label = 'ATT model (advanced)')
plot5 = ax.plot(x_points_data+0.03,effort_cat[2]*100,'v',alpha = .6, color='black', label = 'ATT model (expert)')
ax.set_ylabel(r'Effect on effort (in pp)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
#ax.legend(fontsize = 13)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6),fontsize=12,ncol=2)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/effect_effort.pdf', format='pdf')



"""
#ATT data vs ATT model
fig, ax=plt.subplots()
plot1 = ax.plot(x_points_data,inter_data,'bo',alpha=.7,label = 'ATT data ('+'{:04.2f}'.format(att_data) + r'$\sigma$)')
plot2 = ax.errorbar(x_points_data, inter_data, yerr=se_data, fmt='none', color='k')
plot3 = ax.plot(x_points,y_sim_2,'--o',alpha = .8, color='sandybrown', label = 'ATT model ('+'{:04.2f}'.format(att_mean_sim) + r'$\sigma$)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
ax.legend(loc = 'upper left',fontsize = 13)
#ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1),fontsize=12,ncol=3)
plt.tight_layout()
plt.show()
#fig.savefig('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/counterfactual1_v2.pdf', format='pdf')
"""

"""
#ATT predicted data vs predicted ATT model
data_reg['XY_distance_2'] = data_reg['XY_distance']**2
data_reg['inter_xy'] = data_reg['XY_distance']*data_reg['inter']
data_reg['inter_xy_2'] = data_reg['XY_distance_2']*data_reg['inter']

data_reg.reset_index(inplace = True)
data_reg.set_index(['rbd', 'agno'],inplace = True)
model_reg = PanelOLS(dependent = data_reg['stdsimce'], 
                         exog = data_reg[['constant', 'd_trat', 'd_year',
                                            'XY_distance_2','inter_xy','inter_xy_2','edp', 
                                            'edm', 'ingreso', 'experience']], entity_effects = True)

results = model_reg.fit(cov_type='clustered', clusters=data_reg['drun'])
y_data_hat = results.params.inter_xy*x_points + results.params.inter_xy_2*x_points**2


fig, ax=plt.subplots()
plot1 = ax.plot(x_points,y_data_hat,'bo',alpha=.7,label = 'ATT data ('+'{:04.2f}'.format(att_data) + r'$\sigma$)')
#plot2 = ax.errorbar(x_points_data, inter_data, yerr=se_data, fmt='none', color='k')
plot3 = ax.plot(x_points,y_sim_2,'--o',alpha = .8, color='sandybrown', label = 'ATT model ('+'{:04.2f}'.format(att_mean_sim) + r'$\sigma$)')
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Distance to nearest cutoff', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(0,0.26)
ax.legend(loc = 'upper left',fontsize = 13)
#ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1),fontsize=12,ncol=3)
plt.tight_layout()
plt.show()
"""

