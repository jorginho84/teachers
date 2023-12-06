"""
exec(open("/home/jrodriguezo/teachers/codes/table.py").read())


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
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")

#Betas and var-cov matrix
se_vector = np.load("/home/jrodriguezo/teachers/results/se_model_v54.npy")
betas_nelder = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v54.npy")

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

#Effort teachers (this is t_test function in utility class)
#portfolio
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



#This is for the paper
with open('/home/jrodriguezo/teachers/results/estimates.tex','w') as f:
    f.write(r'\begin{tabular}{lcccc}'+'\n')
    f.write(r'\hline' + '\n')
    f.write(r'Parameter &  & Estimate & & S.E. \bigstrut\\' + '\n')
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


