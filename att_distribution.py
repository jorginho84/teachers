# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:06 2021

This .py generate the density graph of ATT 

@author: pjac2
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
sys.path.append("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_profe_fit")
#import gridemax
import time
#import int_linear
import between
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
from utility_counterfactual_exp import Count_2
import simdata_c as sdc
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


np.random.seed(123)

#Betas and var-cov matrix

#betas_nelder = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/betasopt_model_v23.npy")
#df = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/data_main_regmain_v2023.dta')
#moments_vector = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/moments.npy")
#ses_opt = np.load('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/ses_model.npy')
#data = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGITnewmodel/data_pythonpast_v2023.dta')

betas_nelder  = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_profe_fit/betasopt_model_v2023.npy")

data_1 = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_profe_fit/data_pythonpast.dta')

data = data_1[data_1['d_trat']==1]

N = np.array(data['experience']).shape[0]

n_sim = 100

simce = []
baseline_p = []
income = [] 




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
    
    model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality, priotity_aep)
    
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
    



print ('')
print ('ATT equals ', np.mean(simce[1] - simce[0]))
print ('')


#For validation purposes
att_sim = simce[1] - simce[0]
att_cost = income[1] - income[0]
att_mean_sim = np.mean(att_sim)


#Data complete
data_reg = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_profe_fit/FINALdata.dta')

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

#data_reg["stdsimce"].value_counts()
#data_reg["stdsimce"].info()
#data_reg["experience"].isna().sum()

#aquí restricción nueva de data

"""
keep if stdsimce_m != . & stdsimce_l != .

destring drun_l drun_m, force replace
foreach variable in "eval_year" "drun" "experience" "d_trat" "inter" "d_year"{
	
	gen `variable' = .
	replace `variable' = `variable'_m if `variable'_m == `variable'_l
	replace `variable' = `variable'_m if `variable'_l == . & `variable'_m != .  
	replace `variable' = `variable'_l if `variable'_l != . & `variable'_m == .
	
}

local S_controls edp edm ingreso
local T_controls experience

*fixing sample size for all models
foreach variable in `S_controls' `T_controls' drun d_trat d_year inter{
	keep if `variable' != .
}

eval_year == 1 | eval_year == 2018 | eval_year == 0

"""

"""
#drop if eval16==1 & eval17==1
data_reg_1 = data_reg[(data_reg['eval16']==1) & (data_reg['eval17']==1)].index
data_reg.drop(data_reg_1, inplace = True)
#drop if eval16==1 & eval18==1
data_reg_2 = data_reg[(data_reg['eval16']==1) & (data_reg['eval18']==1)].index
data_reg.drop(data_reg_2, inplace = True)
"""

#REG Stata
#reg stdsimce_m d_trat d_year inter if (eval_year == 1 | eval_year == 2018 | eval_year == 0), vce(cluster drun)
#data_reg = data_reg[ (data_reg['eval_year'] == 1) | (data_reg['eval_year'] == 2018) | (data_reg['eval_year'] == 0) ]

y = np.array(data_reg['stdsimce'])
x_1 = np.array(data_reg['d_trat'])
x_2 = np.array(data_reg['d_year'])
x_3 = np.array(data_reg['inter'])
x = np.transpose(np.array([x_1, x_2, x_3]))
x = sm.add_constant(x)
cov_drun = np.array(data_reg['drun'])

model_reg = sm.OLS(exog=x, endog=y)
results = model_reg.fit(cov_type='cluster', cov_kwds={'groups': cov_drun}, use_t=True)

#reg_data = sm.OLS("stdsimce_m ~ d_trat + d_year + inter", data_reg).fit(cov_type='cluster', cov_kwds={'groups': data_reg['drun']})

print(results.summary())

#Recover the data

inter_data = results.params[3].round(8)
error_data = results.bse[3].round(8)
number_obs = results.nobs
inter_posit = inter_data + 1.96 * np.sqrt(results.normalized_cov_params[3,3])
inter_negat = inter_data - 1.96 * np.sqrt(results.normalized_cov_params[3,3])

#Data Graph

#dataf_graph = {'ATTsim': att_sim, 'ATTdata': att_sim}
dataf_graph = {'ATTsim': att_sim}
dataf_graph = pd.DataFrame(dataf_graph, columns=['ATTsim'])


# Graphs Density 

dataf_graph.plot.kde(linewidth=3, legend=False,alpha = 0.6, color = 'blue');
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
#dataf_graph.plot.kde(linewidth=3, legend=False);
plt.axvline(x=att_mean_sim, ymin=0, ymax=0.95, color='sandybrown', linestyle='-', linewidth=2,alpha = 0.8)
plt.axvline(x=inter_data, ymin=0, ymax=0.95, color='black', linestyle='-', linewidth=2,alpha = 0.6)
plt.axvline(x=inter_posit, ymin=0, ymax=0.95, color='black', linestyle='--', linewidth=1.3,alpha = 0.6)
plt.axvline(x=inter_negat, ymin=0, ymax=0.95, color='black', linestyle='--', linewidth=1.3,alpha = 0.6)
plt.annotate("Simulated ATT" "\n" + "("   +'{:04.2f}'.format(att_mean_sim) + r"$\sigma$s)", xy=(0.08, 1),
            xytext=(0.32, 1), arrowprops=dict(arrowstyle="->"))
plt.annotate("Data ATT" "\n" + "(" +'{:04.2f}'.format(inter_data) + r"$\sigma$s)", xy=(0.018, 1),
            xytext=(-0.4, 1), arrowprops=dict(arrowstyle="->"))
plt.annotate("Treatment effects distribution", xy=(0.2, 1.7),
            xytext=(0.2, 2), arrowprops=dict(arrowstyle="->"))
plt.xlim(-0.6,0.6)
plt.savefig('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Result/att_distribution_v2023_v3.pdf', format='pdf')




