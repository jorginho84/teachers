"""
exec(open("/home/jrodriguezo/teachers/codes/estimates_table.py").read())


This file stores the estimated coefficients in a .tex table


"""

from __future__ import division #omit for python 3.x
import numpy as np
import pandas as pd
import itertools
import sys, os
from joblib import Parallel, delayed
from scipy import stats
from scipy import interpolate
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import time
import openpyxl
sys.path.append("/home/jrodriguezo/teachers/codes")
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd


#Betas and var-cov matrix
se_vector = np.load("/home/jrodriguezo/teachers/results/se_model_v56.npy")
betas_nelder = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v56.npy")

#Utility function
gamma_0 = betas_nelder[14]
gamma_1 = betas_nelder[15]
gamma_2 = betas_nelder[16]

se_gamma_0 =se_vector[14]
se_gamma_1 =se_vector[15]
se_gamma_2 =se_vector[16]



#Student HC
betas_opt_t = np.array([betas_nelder[8],betas_nelder[9],betas_nelder[10],
	betas_nelder[11],betas_nelder[12],betas_nelder[13]]).reshape((6,1))

se_betas_opt_t=np.array([se_vector[8],se_vector[9],se_vector[10],
                         se_vector[11],se_vector[12],se_vector[14]]).reshape((6,1))

#Teacher test scores
alphas_port = np.array([betas_nelder[0],betas_nelder[1],betas_nelder[2], betas_nelder[3]]).reshape((4,1))
se_alphas_port = np.array([se_vector[0],se_vector[1],se_vector[2],se_vector[3]]).reshape((4,1))

alphas_test = np.array([betas_nelder[4],betas_nelder[5],betas_nelder[6],betas_nelder[7]]).reshape((4,1))
se_alphas_test = np.array([se_vector[4],se_vector[5],se_vector[6],se_vector[7]]).reshape((4,1))


###########.TEX table##################

utility_list_beta = [gamma_0,gamma_1,gamma_2]
utility_list_se = [se_gamma_0,se_gamma_1,se_gamma_2]
utility_names = [r'Teaching skills (Portfolio) effort ($\gamma_1$)',r'Subject knowledge (STEI) effort ($\gamma_1$)',r'Preference for student performance ($\gamma_h$)']

beta_list_beta = [betas_opt_t[0,0],betas_opt_t[1,0],betas_opt_t[2,0],betas_opt_t[4,0],betas_opt_t[5,0],betas_opt_t[3,0]]
beta_list_se = [se_betas_opt_t[0,0],se_betas_opt_t[1,0],
se_betas_opt_t[2,0],se_betas_opt_t[4,0],se_betas_opt_t[5,0],se_betas_opt_t[3,0]]
wage_names = [r'Constant ($\beta_0$)',r'Portfolio effort ($\beta_1$)',r'STEI effort ($\beta_2$)',r'Experience effect ($\beta_3$)',r'Past performance ($\beta_4$)',r'SD of shock ($ \sigma_\nu$)']


alphas_port_list= [alphas_port[0,0],alphas_port[1,0],alphas_port[3,0],alphas_port[2,0]]
se_alphas_port_list = [se_alphas_port[0,0],se_alphas_port[1,0],se_alphas_port[3,0],se_alphas_port[2,0]]
prod_names_young = [r'Effort ($\alpha_1^p$)',r'Experience ($\alpha_2^p$)', r'Past performance ($\alpha_3^p$)',r'SD of shock ($\sigma_{M_p}$)']

alphas_test_list = [alphas_test[0,0],alphas_test[1,0],alphas_test[3,0],alphas_test[2,0]]
se_alphas_test_list = [se_alphas_test[0,0],se_alphas_test[1,0],se_alphas_test[3,0],se_alphas_test[2,0]]
prod_names_old = [r'Effort ($\alpha_1^s$)',r'Experience ($\alpha_2^s$)', r'Past performance ($\alpha_3^s$)',r'SD of shock ($\sigma_{M_s}$)']


with open('/home/jrodriguezo/teachers/results/estimates.tex','w') as f:
    f.write(r'\begin{tabular}{lcccc}'+'\n')
    f.write(r'\hline' + '\n')
    f.write(r'Parameter &  & Estimate & & S.E.' + '\n')
    f.write(r'\hline' + '\n')
    f.write(r'\emph{A. Utility function  }  &       &       &       &  \\' + '\n')
    for j in range(len(utility_list_beta)):
        if (j == 0) | (j == 2):
            f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
             r' &  & '+ '{:1.2E}'.format(utility_list_se[j])+r' \\' + '\n')
        else:
            f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
             r' &  & '+ '{:04.3f}'.format(utility_list_se[j])+r' \\' + '\n')
        
    f.write(r' &       &       &       &  \\' + '\n')
    f.write(r'\emph{B. Production function of SIMCE} &       &       &       &  \\' + '\n')
    
    for j in range(len(beta_list_beta)):
        if j == 4:
            f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(beta_list_beta[j]) + 
                r' &  & '+ '{:01.2E}'.format(beta_list_se[j])+r' \\' + '\n')

        elif j == 5:
            f.write(wage_names[j]+r' &  &  '+ '{:01.2E}'.format(beta_list_beta[j]) + 
                r' &  & '+ '{:01.2E}'.format(beta_list_se[j])+r' \\' + '\n')

        else:
            f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(beta_list_beta[j]) + 
               r' &  & '+ '{:04.3f}'.format(beta_list_se[j])+r' \\' + '\n')
    
    f.write(r' &       &       &       &  \\' + '\n')
    f.write(r'\emph{C. Production function of Portfolio} &       &       &       &  \\' + '\n')
    
    for j in range(len(alphas_port_list)):
        f.write(prod_names_young[j]+r' &  &  '+ '{:04.3f}'.format(alphas_port_list[j]) +
            r' &  & '+ '{:04.3f}'.format(se_alphas_port_list[j])+r' \\' + '\n')
        
    f.write(r' &       &       &       &  \\' + '\n')
    f.write(r'\emph{D. Production function of STEI} &       &       &       &  \\' + '\n')
    
    for j in range(len(alphas_test_list)):
        f.write(prod_names_old[j] + r' &  &  '+ '{:04.3f}'.format(alphas_test_list[j]) +
                        r' &  & '+ '{:04.3f}'.format(se_alphas_test_list[j])+r' \\' + '\n')
    f.write(r'\hline'+'\n')
    f.write(r'\end{tabular}' + '\n')
    f.close()
    



#################################################################
#Interpreting effects in teachers' test scores
data = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')

#Portfolio
delta_p = ((1/(1+np.exp(alphas_port[0,0]))) + (1/3))*3
delta_t = ((1/(1+np.exp(alphas_test[0,0]))) + (1/3))*3
    
# TREATMENT #
treatment = np.array(data['d_trat'])

# EXPERIENCE #
years = np.array(data['experience'])

# SCORE PORTFOLIO #
p1_0 = np.array(data['score_port_past'])
p1 = np.array(data['score_port'])

# SCORE TEST #
p2_0 = np.array(data['score_test_past'])
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

AEP_priority = np.array(data['priority_aep'])

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
    


# basic rent by hour in dollar (average mayo 2020, until 13/05/2020) *
# value hour (pesos)= 14403 *
# value hour (pesos)= 15155 *

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

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,
                     TrameI,priotity,rural_rbd,locality, AEP_priority)

modelSD = sd.SimData(N,model)

def prod_fn(effort):

    p1v1_past = np.where(np.isnan(p1_0), 0, p1_0)
    p2v1_past = np.where(np.isnan(p2_0), 0, p2_0)
        
     
    p0_past = np.zeros(p1v1_past.shape)
    p0_past = np.where((p1v1_past == 0),p2v1_past, p0_past)
    p0_past = np.where((p2v1_past == 0),p1v1_past, p0_past)
    p0_past = np.where((p1v1_past != 0) & (p2v1_past != 0) ,(p1_0 + p2_0)/2, p0_past)
    p0_past = (p0_past-np.mean(p0_past))/np.std(p0_past)
    p0_past[treatment == 0] = 0
        
    d_effort_t1 = effort == 1
    d_effort_t2 = effort == 2
    d_effort_t3 = effort == 3
        
    effort_m = d_effort_t1 + d_effort_t3
    effort_h = d_effort_t2 + d_effort_t3
    pb = []
    
    for j in range(2):
        pb.append(alphas[j][0] + alphas[j][1]*effort_m + alphas[j][2]*effort_h + alphas[j][3]*years/10 + alphas[j][5]*p0_past)
    
    p_scores = [((1/(1+np.exp(-pb[0]))) + (1/3))*3, ((1/(1+np.exp(-pb[1]))) + (1/3))*3]

    return [p_scores, pb]


n_sims = 100
portfolio_pot = np.zeros((N,n_sims))
stei_pot = np.zeros((N,n_sims))

for i in range(1,n_sims):
    opt = modelSD.choice()
    effort = opt['Opt Effort']
    p_scores_list = prod_fn(effort)
    portfolio_pot[:,i] = p_scores_list[1][0]
    stei_pot[:,i] = p_scores_list[1][1]


portfolio_base = np.mean(portfolio_pot,axis= 1)
stei_base = np.mean(stei_pot,axis= 1)

delta_port = 3*((1+np.exp(-portfolio_base))**(-2))*np.exp(-portfolio_base)*alphas[0][1]
delta_stei = 3*((1+np.exp(-stei_base))**(-2))*np.exp(-stei_base)*alphas[1][2]



            




