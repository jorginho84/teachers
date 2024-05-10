"""
This code computes effects on simce of different PFP policies

exec(open("/home/jrodriguezo/teachers/codes/wtp_pfp_slopes.py").read())
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
import simdata_c as sdc
import estimate as est
from utility_counterfactual_att import Count_att_2
from utility_counterfactual_att_categories import Count_att_2_cat
from utility_counterfactual_att_pfp import Count_att_2_pfp
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


betas_nelder  = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v56.npy")

#Only treated teachers
#data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data_1 = pd.read_stata(r'/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')
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

alphas = [[0, betas_nelder[0],0,betas_nelder[1],
             betas_nelder[2], betas_nelder[3]],
            [0, 0,betas_nelder[4],betas_nelder[5],
            betas_nelder[6], betas_nelder[7]]]
            
betas = [betas_nelder[8], betas_nelder[9], betas_nelder[10],betas_nelder[11],betas_nelder[12],betas_nelder[13]]
gammas = [betas_nelder[14],betas_nelder[15],betas_nelder[16]]

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

b_baseline = 900
a_baseline = 1000
c_baseline = 0.5
#List of choices across counterfactuals

#baseline
simce = []
baseline_p = []
income = []
effort_p = []
effort_t = []
wtp_list = []
utils_list = []
delta_income = []
delta_simce = []
portfolio_list = []
test_list = []

#Two series of outcomes: across b and c
blen = len(np.arange(0,3000,500))
clen = len(np.arange(0, 1, 0.05))
simce_a = np.zeros((N,blen,clen))
baseline_p_a = np.zeros((N,blen,clen))
income_a = np.zeros((N,blen,clen))
effort_p_a = np.zeros((N,blen,clen))
effort_t_a = np.zeros((N,blen,clen))
wtp_list_a = np.zeros((N,blen,clen))
utils_list_a = np.zeros((N,blen,clen))
delta_income_a = np.zeros((N,blen,clen))
delta_simce_a = np.zeros((N,blen,clen))
portfolio_list_a = np.zeros((N,blen,clen))
test_list_a = np.zeros((N,blen,clen))

#Original ATT
for x in range(0,2):
    
    # TREATMENT #

   if x == 0:
        treatment = np.ones(N)*x
         #Original STPD
        model = Count_att_2(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality, priotity_aep)
        
        simce_sims = np.zeros((N,n_sim))
        income_sims = np.zeros((N,n_sim))
        effort_p_sims = np.zeros((N,n_sim))
        effort_t_sims = np.zeros((N,n_sim))
        utils_sims = np.zeros((N,n_sim))
        portfolio_sims = np.zeros((N,n_sim))
        test_sims = np.zeros((N,n_sim))
        placement_sims = np.zeros((N,n_sim))
    
    
        modelSD = sd.SimData(N,model)
        
            
        for j in range(n_sim):
            opt = modelSD.choice()
            utils_sims[:,j] = opt['Opt Utility']
            simce_sims[:,j] = opt['Opt Simce']
            portfolio_sims[:,j] = opt['Opt Teacher'][0]
            test_sims[:,j] = opt['Opt Teacher'][1]
            placement_sims[:,j] = opt['Opt Placement'][0]
            income_sims[:,j] = opt['Opt Income'][1]
        
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
        effort_p.append(np.mean(effort_p_sims,axis = 1))
        effort_t.append(np.mean(effort_t_sims,axis = 1))
        utils_list.append(np.mean(utils_sims,axis = 1))
        portfolio_list.append(np.mean(portfolio_sims,axis = 1))
        test_list.append(np.mean(test_sims,axis = 1))
        
        
   if x == 1:
      treatment = np.ones(N)
      c_count = 0
      for c in np.arange(0, 1, 0.05):
            
         b_count = 0
         for b in range(0,3000,500):

            model = Count_att_2_pfp(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                                priotity,rural_rbd,locality, priotity_aep,a_baseline,b,c)
            
            simce_sims = np.zeros((N,n_sim))
            income_sims = np.zeros((N,n_sim))
            effort_p_sims = np.zeros((N,n_sim))
            effort_t_sims = np.zeros((N,n_sim))
            utils_sims = np.zeros((N,n_sim))
            portfolio_sims = np.zeros((N,n_sim))
            test_sims = np.zeros((N,n_sim))
            placement_sims = np.zeros((N,n_sim))
        
        
            modelSD = sdc.SimDataC(N,model)
            
                
            for j in range(n_sim):
                opt = modelSD.choice()
                utils_sims[:,j] = opt['Opt Utility']
                simce_sims[:,j] = opt['Opt Simce']
                portfolio_sims[:,j] = opt['Opt Teacher'][0]
                test_sims[:,j] = opt['Opt Teacher'][1]
                placement_sims[:,j] = opt['Opt Placement'][0]
                income_sims[:,j] = opt['Opt Income'][0]
            
                effort_v1 = opt['Opt Effort']
                d_effort_t1 = effort_v1 == 1
                d_effort_t2 = effort_v1 == 2
                d_effort_t3 = effort_v1 == 3
                
                effort_m = d_effort_t1 + d_effort_t3
                effort_h = d_effort_t2 + d_effort_t3
                effort_p_sims[:,j] = effort_m
                effort_t_sims[:,j] = effort_h
                
            simce_a[:,b_count,c_count] = np.mean(simce_sims,axis=1)
            income_a[:,b_count,c_count] = np.mean(income_sims,axis=1)
            effort_p_a[:,b_count,c_count] = np.mean(effort_p_sims,axis = 1)
            effort_t_a[:,b_count,c_count] = np.mean(effort_t_sims,axis = 1)
            utils_list_a[:,b_count,c_count] = np.mean(utils_sims,axis = 1)
            portfolio_list_a[:,b_count,c_count] = np.mean(portfolio_sims,axis = 1)
            test_list_a[:,b_count,c_count] = np.mean(test_sims,axis = 1)

            b_count = b_count + 1

         c_count = c_count + 1

#Parameters for MVPF
rho = 0.1
tax = 0.35
#Averge annual wage (2020 dollars, jan2020-jan2002)
av_annual_wage = 4565*(28310.86/16262.66)

wage = np.zeros(40)
interes = 0.03

for i in range(40):
   wage[i] = av_annual_wage/((1+interes)**(i))
lifetime_earnings = np.sum(wage)

#Effects on SIMCE, effort, and WTP
s_a = np.zeros((blen,clen))
p_a = np.zeros((blen,clen))
t_a = np.zeros((blen,clen))
wtp_a = np.zeros((blen,clen))
inc_a = np.zeros((blen,clen))
wtp_student = np.zeros((blen,clen))
wtp_teachers = np.zeros((blen,clen))
wtp_overall = np.zeros((blen,clen))
provision = np.zeros((blen,clen))
revenue = np.zeros((blen,clen))
net_cost = np.zeros((blen,clen))
mvpf = np.zeros((blen,clen))


for x in range(blen):
   for y in range(clen):
       s_a[x,y] = np.mean(simce_a[:,x,y] - simce[0])
       p_a[x,y] = np.mean(effort_p_a[:,x,y] - effort_p[0])
       t_a[x,y] = np.mean(effort_t_a[:,x,y] - effort_t[0])

       #Changes in WTP
       wtp_teachers[x,y] = np.mean(-1*(np.exp(utils_list_a[:,x,y] - (gammas[0]*effort_p[0] + gammas[1]*effort_t[0] + gammas[2]*simce[0])) - income[0]))

       #Changes in income (to compute added revenues and provision cost)
       inc_a[x,y] = np.mean(income_a[:,x,y] - income[0])

       wtp_student[x,y] = s_a[x,y]*rho*lifetime_earnings
       wtp_overall[x,y] = wtp_student[x,y] + wtp_teachers[x,y]
       provision[x,y] = np.mean(inc_a[x,y])*12
       revenue[x,y] = s_a[x,y]*rho*lifetime_earnings*tax + provision[x,y]*tax
       net_cost[x,y] = provision[x,y] - revenue[x,y]
       mvpf[x,y] = wtp_overall[x,y] / net_cost[x,y]

c_points = np.arange(0, 1, 0.05)
b_points = np.arange(0,3000,500)

fig, ax=plt.subplots()

color_t = 1
for x in range(blen):
   plot1 = ax.scatter(c_points,s_a[x,:],color='b' ,alpha=color_t, label = 'Slope ='+'{:04.2f}'.format(b_points[x]))
   color_t = color_t - 0.15
ax.set_ylabel(r'Effect on SIMCE (in $\sigma$)', fontsize=13)
ax.set_xlabel(r'Parameter in wage schedule (baseline/slope)', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
ax.set_ylim(-0.12,0.12)
#plt.xticks(x, ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-'],fontsize=12)
ax.legend(loc = 'upper left',fontsize = 13)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/simce_pfp_shares_slopes_simce.pdf', format='pdf')
plt.close()


fig, ax=plt.subplots()
color_t = 1
for x in range(blen):
   plot1 = ax.scatter(c_points,mvpf[x,:],color='b' ,alpha=color_t, label = 'Slope ='+'{:04.2f}'.format(b_points[x]))
   color_t = color_t - 0.15
ax.set_ylabel(r'MVPF', fontsize=13)
ax.set_xlabel(r'Parameter in wage schedule (baseline/slope)', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
ax.set_ylim(-0.12,0.12)
#plt.xticks(x, ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-'],fontsize=12)
ax.legend(loc = 'upper left',fontsize = 13)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/simce_pfp_shares_slopes_mvpf.pdf', format='pdf')





