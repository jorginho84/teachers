"""
exec(open("/home/jrodriguez/NH_HC/codes/model_v2/estimation/table.py").read())


This file stores the estimated coefficients in an excel table.

Before running, have var-cov matrix of estimated parameters (se.py in ses folder)


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
se_vector = np.load("//Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/se_model_v4.npy")
betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v13.npy")

#Utility function
gamma_0 = betas_nelder[14]
gamma_1 = betas_nelder[15]
gamma_2 = betas_nelder[16]


se_gamma_0 =se_vector[14]
se_gamma_1 =se_vector[15]
se_gamma_2 =se_vector[16]


#Effort Student (this is t_student function in utility class)
betas_opt_t = np.array([betas_nelder[10],betas_nelder[11],betas_nelder[12],
	betas_nelder[13]]).reshape((4,1))

se_betas_opt_t=np.array([se_vector[10],se_vector[11],se_vector[12],se_vector[13]]).reshape((4,1))

#Effort teachers (this is t_test function in utility class)
#portfolio
alphas_port = np.array([betas_nelder[0],betas_nelder[1],betas_nelder[2],
               betas_nelder[3],betas_nelder[4]]).reshape((6,1))

se_alphas_port = np.array([se_vector[0],se_vector[1],se_vector[2],
               se_vector[3],se_vector[4]]).reshape((6,1))


alphas_test = np.array([betas_nelder[5],betas_nelder[6],betas_nelder[7],
               betas_nelder[8],betas_nelder[9]]).reshape((6,1))

se_alphas_test = np.array([se_vector[5],se_vector[6],se_vector[7],
               se_vector[8],se_vector[9]]).reshape((6,1))




###########.TEX table##################

utility_list_beta = [gamma_0,gamma_1,gamma_2]
utility_list_se = [se_gamma_0,se_gamma_1,se_gamma_2]
utility_names = [r'Portfolio effort',r'PKT effort',r'Preference for student performance']

beta_list_beta = [betas_opt_t[0,0],betas_opt_t[1,0],betas_opt_t[2,0],betas_opt_t[3,0]]
beta_list_se = [se_betas_opt_t[0,0],se_betas_opt_t[1,0],
se_betas_opt_t[2,0],se_betas_opt_t[3,0]]
wage_names = ['Constant', 'Portfolio effort','PKT effort','Measurement error']


alphas_port_list= [alphas_port[0,0],alphas_port[1,0],alphas_port[2,0],alphas_port[3,0]]
se_alphas_port_list = [se_alphas_port[0,0],se_alphas_port[1,0],se_alphas_port[2,0],se_alphas_port[3,0]]
prod_names_young = ['Constant', 'Effort','Experience', r'Variance of shock', r'Past performance']

alphas_test_list = [alphas_test[0,0],alphas_test[1,0],alphas_test[2,0],alphas_test[3,0]]
se_alphas_test = [se_alphas_test[0,0],se_alphas_test[1,0],se_alphas_test[2,0],se_alphas_test[3,0]]
prod_names_old = ['Constant', 'Effort',
                    'Experience', r'Variance of shock', r'Past performance']



#This is for the paper
with open('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/estimates.tex','w') as f:
	f.write(r'\begin{tabular}{lcccc}'+'\n')
	f.write(r'\hline' + '\n')
	f.write(r'Parameter &  & Estimate & & S.E. \bigstrut\\' + '\n')
	f.write(r'\hline' + '\n')
	f.write(r'\emph{A. Utility function  }  &       &       &       &  \\' + '\n')
	for j in range(len(utility_list_beta)):
		f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(utility_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{B. Production function of SIMCE} &       &       &       &  \\' + '\n')
	for j in range(len(beta_list_beta)):
		f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(beta_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(beta_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{C. Production function of PPCP} &       &       &       &  \\' + '\n')
	for j in range(len(alphas_port_list)):
		f.write(prod_names_young[j]+r' &  &  '+ '{:04.3f}'.format(alphas_port_list[j]) +
			r' &  & '+ '{:04.3f}'.format(se_alphas_port_list[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{D. Production function of STEI} &       &       &       &  \\' + '\n')
	for j in range(len(alphas_test_list)):
		f.write(prod_names_old[j] + r' &  &  '+ '{:04.3f}'.format(alphas_test_list[j]) +
			r' &  & '+ '{:04.3f}'.format(se_alphas_test[j])+r' \\' + '\n')


	f.write(r'\hline'+'\n')
	f.write(r'\end{tabular}' + '\n')
	f.close()


