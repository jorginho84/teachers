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
sys.path.append("D:\Git\TeachersMaster")

#Betas and var-cov matrix
se_vector = np.load("D:\Git\TeachersMaster\se_model.npy")
betas_nelder  = np.load("D:\\Git\\TeachersMaster\\betasopt_model.npy")

#Utility function
gamma_0 = betas_nelder[14]
gamma_1 = betas_nelder[15]
gamma_2 = betas_nelder[16]
mu_c = -0.56
mu_theta = 0.5


se_gamma_0 =se_vector[14]
se_gamma_1 =se_vector[15]
se_gamma_2 =se_vector[16]


#Effort Student (this is t_student function in utility class)
betas_opt_t = np.array([betas_nelder[11],betas_nelder[12],
	betas_nelder[13]]).reshape((3,1))

se_betas_opt_t=np.array([se_vector[11],se_vector[12],se_vector[13]]).reshape((3,1))


#income_male_betas = np.array([betas_nelder[8],betas_nelder[9],
	#betas_nelder[10]]).reshape((3,1))
#c_emp_spouse = betas_nelder[11]


#sigma_income_male_betas = np.array([se_vector[8],se_vector[9],
	#se_vector[10]]).reshape((3,1))
#sigma_c_emp_spouse = se_vector[11]

#Effort teachers (this is t_test function in utility class)
#portfolio
alphas_port = np.array([betas_nelder[0],betas_nelder[1],betas_nelder[2],
               betas_nelder[3],betas_nelder[4]]).reshape((5,1))

se_alphas_port = np.array([se_vector[0],se_vector[1],se_vector[2],
               se_vector[3],se_vector[4]]).reshape((5,1))


alphas_test = np.array([betas_nelder[5],betas_nelder[6],betas_nelder[7],
               betas_nelder[8],betas_nelder[9]]).reshape((5,1))

se_alphas_test = np.array([se_vector[5],se_vector[6],se_vector[7],
               se_vector[8],se_vector[9]]).reshape((5,1))

#gamma2 = [betas_nelder[13],betas_nelder[16]]
#gamma3 = [betas_nelder[14],betas_nelder[17]]
#tfp = betas_nelder[18]
#sigma2theta = betas_nelder[19]

#se_gamma1 = [se_vector[12],se_vector[15]]
#se_gamma2 = [se_vector[13],se_vector[16]]
#se_gamma3 = [se_vector[14],se_vector[17]]
#se_tfp = se_vector[18]
#se_sigma2theta = se_vector[19]


kappas = [0,0]

sigma_z  = [0,0]

#rho_theta_epsilon = betas_nelder[20]
#se_rho_theta_epsilon = se_vector[20]

#qprob = betas_nelder[21]
#se_qprob = se_vector[21]

#First measure is normalized. starting arbitrary values
lambdas=[1,1]



###########.TEX table##################

utility_list_beta = [gamma_0,gamma_1,gamma_2]
utility_list_se = [se_gamma_0,se_gamma_1,se_gamma_2]
utility_names = [r'gamma_0',r'gamma_1',r'gamma_2']

beta_list_beta = [betas_opt_t[0,0],betas_opt_t[1,0],betas_opt_t[2,0]]
beta_list_se = [se_betas_opt_t[0,0],se_betas_opt_t[1,0],
se_betas_opt_t[2,0]]
wage_names = ['beta_0', 'beta_1','beta_2']

#swage_list_beta = [income_male_betas[0,0],income_male_betas[1,0],income_male_betas[2,0],
#c_emp_spouse]
#swage_list_se = [sigma_income_male_betas[0,0],sigma_income_male_betas[1,0],sigma_income_male_betas[2,0],
#sigma_c_emp_spouse]
#swage_names = ['High school dummy', 'Constant', 'SD of error term','Employment probability']


alphas_port_list= [alphas_port[0,0],alphas_port[1,0],alphas_port[2,0],alphas_port[3,0],alphas_port[4,0]]
se_alphas_port_list = [se_alphas_port[0,0],se_alphas_port[1,0],se_alphas_port[2,0],se_alphas_port[3,0],se_alphas_port[4,0]]
prod_names_young = [r'alpha_00', r'alpha_01', r'alpha_02', r'alpha_03', r'alpha_04']

alphas_test_list = [alphas_test[0,0],alphas_test[1,0],alphas_test[2,0],alphas_test[3,0],alphas_test[4,0]]
se_alphas_test = [se_alphas_test[0,0],se_alphas_test[1,0],se_alphas_test[2,0],se_alphas_test[3,0],se_alphas_test[4,0]]
prod_names_old = [r'alpha_10', r'alpha_11', r'alpha_12', r'alpha_13', r'alpha_14']


#prod_fn_common = [sigma2theta,rho_theta_epsilon,qprob]
#prod_fn_common_se = [se_sigma2theta,se_rho_theta_epsilon,se_qprob]

#prod_fn_names_common = [r'SD of shock to human capital', r'Correlation of wage and initial human capital'
#, r'Probability of free child care']

#This is for the paper
with open('D:\Git\TeachersMaster/estimates.tex','w') as f:
	f.write(r'\begin{tabular}{lcccc}'+'\n')
	f.write(r'\hline' + '\n')
	f.write(r'Parameter &  & Estimate & & S.E. \bigstrut\\' + '\n')
	f.write(r'\hline' + '\n')
	f.write(r'\emph{A. Utility function} &       &       &       &  \\' + '\n')
	for j in range(len(utility_list_beta)):
		f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(utility_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{B. Effort Student} &       &       &       &  \\' + '\n')
	for j in range(len(beta_list_beta)):
		f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(beta_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(beta_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{C. Effort teachers Portfolio} &       &       &       &  \\' + '\n')
	for j in range(len(alphas_port_list)):
		f.write(prod_names_young[j]+r' &  &  '+ '{:04.3f}'.format(alphas_port_list[j]) +
			r' &  & '+ '{:04.3f}'.format(se_alphas_port_list[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{D. Effort teachers Test} &       &       &       &  \\' + '\n')
	for j in range(len(alphas_test_list)):
		f.write(prod_names_old[j] + r' &  &  '+ '{:04.3f}'.format(alphas_test_list[j]) +
			r' &  & '+ '{:04.3f}'.format(se_alphas_test[j])+r' \\' + '\n')

	#for j in range(len(prod_list_beta_old)):
		#f.write(prod_names_old[j]  + ' (old)' + r' &  &  '+ '{:04.3f}'.format(prod_list_beta_old[j]) +
			#r' &  & '+ '{:04.3f}'.format(prod_list_se_old[j])+r' \\' + '\n')
	#for j in range(len(prod_fn_common)):
		#f.write(prod_fn_names_common[j]+r' &  &  '+ '{:04.3f}'.format(prod_fn_common[j]) +
			#r' &  & '+ '{:04.3f}'.format(prod_fn_common_se[j])+r' \\' + '\n')

	f.write(r'\hline'+'\n')
	f.write(r'\end{tabular}' + '\n')
	f.close()


