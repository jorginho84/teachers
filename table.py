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

#Betas and var-cov matrix
betas_nelder=np.load('/home/jrodriguez/NH_HC/results/Model/estimation/betas_modelv66.npy')
var_cov=np.load('/home/jrodriguez/NH_HC/results/model_v2/estimation/sesv3_modelv66.npy')
se_vector  = np.sqrt(np.diagonal(var_cov))

#Utility function
eta = betas_nelder[0]
alpha_p = betas_nelder[1]
alpha_f = betas_nelder[2]
mu_c = -0.56
mu_theta = 0.5


sigma_eta_opt=se_vector[0]
sigma_alpha_p_opt=se_vector[1]
sigma_alpha_f_opt=se_vector[2]


#wage process
wagep_betas = np.array([betas_nelder[3],betas_nelder[4],
	betas_nelder[5],
	betas_nelder[6],betas_nelder[7]]).reshape((5,1))

sigma_wagep_betas=np.array([se_vector[3],se_vector[4],se_vector[5],
	se_vector[6],se_vector[7]]).reshape((5,1))


income_male_betas = np.array([betas_nelder[8],betas_nelder[9],
	betas_nelder[10]]).reshape((3,1))
c_emp_spouse = betas_nelder[11]


sigma_income_male_betas = np.array([se_vector[8],se_vector[9],
	se_vector[10]]).reshape((3,1))
sigma_c_emp_spouse = se_vector[11]

#Production function [young[cc0,cc1],old]
gamma1 = [betas_nelder[12],betas_nelder[15]]
gamma2 = [betas_nelder[13],betas_nelder[16]]
gamma3 = [betas_nelder[14],betas_nelder[17]]
tfp = betas_nelder[18]
sigma2theta = betas_nelder[19]

se_gamma1 = [se_vector[12],se_vector[15]]
se_gamma2 = [se_vector[13],se_vector[16]]
se_gamma3 = [se_vector[14],se_vector[17]]
se_tfp = se_vector[18]
se_sigma2theta = se_vector[19]


kappas = [0,0]

sigma_z  = [0,0]

rho_theta_epsilon = betas_nelder[20]
se_rho_theta_epsilon = se_vector[20]

qprob = betas_nelder[21]
se_qprob = se_vector[21]

#First measure is normalized. starting arbitrary values
lambdas=[1,1]



###########.TEX table##################

utility_list_beta = [alpha_p,alpha_f,eta]
utility_list_se = [sigma_alpha_p_opt,sigma_alpha_f_opt,sigma_eta_opt]
utility_names = [r'Part-time disutility',
r'Full-time disutility',
r'Preference for human capital']

wage_list_beta = [wagep_betas[0,0],wagep_betas[1,0],wagep_betas[2,0],
wagep_betas[3,0],wagep_betas[4,0]]
wage_list_se = [sigma_wagep_betas[0,0],sigma_wagep_betas[1,0],
sigma_wagep_betas[2,0],sigma_wagep_betas[3,0],sigma_wagep_betas[4,0]]
wage_names = ['High school dummy', 'Trend','Constant', 'Variance of error term','AR(1) error term']

swage_list_beta = [income_male_betas[0,0],income_male_betas[1,0],income_male_betas[2,0],
c_emp_spouse]
swage_list_se = [sigma_income_male_betas[0,0],sigma_income_male_betas[1,0],sigma_income_male_betas[2,0],
sigma_c_emp_spouse]
swage_names = ['High school dummy', 'Constant', 'SD of error term','Employment probability']


prod_list_beta_young = [tfp,gamma1[0],gamma2[0],gamma3[0]]
prod_list_se_young  = [se_tfp,se_gamma1[0],se_gamma2[0],se_gamma3[0]]
prod_names_young = [r'Child care TFP', r'Lagged human capital', r'Income'
, r'Time']

prod_list_beta_old = [gamma1[1],gamma2[1],gamma3[1]]
prod_list_se_old  = [se_gamma1[0],se_gamma2[0],se_gamma3[0]]
prod_names_old = [r'Lagged human capital', r'Income'
, r'Time']


prod_fn_common = [sigma2theta,rho_theta_epsilon,qprob]
prod_fn_common_se = [se_sigma2theta,se_rho_theta_epsilon,se_qprob]

prod_fn_names_common = [r'SD of shock to human capital', r'Correlation of wage and initial human capital'
, r'Probability of free child care']

#This is for the paper
with open('/home/jrodriguez/NH_HC/results/model_v2/estimation/estimates.tex','w') as f:
	f.write(r'\begin{tabular}{lcccc}'+'\n')
	f.write(r'\hline' + '\n')
	f.write(r'Parameter &  & Estimate & & S.E. \bigstrut\\' + '\n')
	f.write(r'\hline' + '\n')
	f.write(r'\emph{A. Utility function} &       &       &       &  \\' + '\n')
	for j in range(len(utility_list_beta)):
		f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(utility_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{B. Wage offer} &       &       &       &  \\' + '\n')
	for j in range(len(wage_list_beta)):
		f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(wage_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(wage_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{C. Spouse income} &       &       &       &  \\' + '\n')
	for j in range(len(swage_list_beta)):
		f.write(swage_names[j]+r' &  &  '+ '{:04.3f}'.format(swage_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(swage_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{D. Production function and child care} &       &       &       &  \\' + '\n')
	for j in range(len(prod_list_beta_young)):
		f.write(prod_names_young[j] + ' (young)' + r' &  &  '+ '{:04.3f}'.format(prod_list_beta_young[j]) +
			r' &  & '+ '{:04.3f}'.format(prod_list_se_young[j])+r' \\' + '\n')

	for j in range(len(prod_list_beta_old)):
		f.write(prod_names_old[j]  + ' (old)' + r' &  &  '+ '{:04.3f}'.format(prod_list_beta_old[j]) +
			r' &  & '+ '{:04.3f}'.format(prod_list_se_old[j])+r' \\' + '\n')
	for j in range(len(prod_fn_common)):
		f.write(prod_fn_names_common[j]+r' &  &  '+ '{:04.3f}'.format(prod_fn_common[j]) +
			r' &  & '+ '{:04.3f}'.format(prod_fn_common_se[j])+r' \\' + '\n')

	f.write(r'\hline'+'\n')
	f.write(r'\end{tabular}' + '\n')
	f.close()


