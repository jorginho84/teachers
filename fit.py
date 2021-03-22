#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:33:39 2021

@author: jorge-home

This code computes fit anaylisis

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
sys.path.append("D:\Git\TeacherBranch")


#Moments Vector


moments_vector = pd.read_excel("D:\Git\TeacherBranch\Outcomes.xlsx", header=3, usecols='C:F').values


#Utility function

simulation_test = np.array([moments_vector[0,1],moments_vector[1,1],moments_vector[2,1],
                            moments_vector[3,1],moments_vector[4,1],moments_vector[17,1]]).reshape((6,1))

simulation_port = np.array([moments_vector[5,1],moments_vector[6,1],moments_vector[7,1],
                            moments_vector[8,1],moments_vector[9,1],moments_vector[18,1]]).reshape((6,1))

simulation_eteacher = np.array([moments_vector[10,1],moments_vector[11,1],moments_vector[12,1],
                                moments_vector[13,1]]).reshape((4,1))

simulation_estudent = np.array([moments_vector[14,1],moments_vector[15,1],moments_vector[16,1]]).reshape((3,1))

mu_c = -0.56
mu_theta = 0.5

##################################

data_test = np.array([moments_vector[0,2],moments_vector[1,2],moments_vector[2,2],
                            moments_vector[3,2],moments_vector[4,2],moments_vector[17,2]]).reshape((6,1))

data_port = np.array([moments_vector[5,2],moments_vector[6,2],moments_vector[7,2],
                            moments_vector[8,2],moments_vector[9,2],moments_vector[18,2]]).reshape((6,1))

data_eteacher = np.array([moments_vector[10,2],moments_vector[11,2],moments_vector[12,2],
                                moments_vector[13,2]]).reshape((4,1))

data_estudent = np.array([moments_vector[14,2],moments_vector[15,2],moments_vector[16,2]]).reshape((3,1))

#################################

error_test = np.array([moments_vector[0,3],moments_vector[1,3],moments_vector[2,3],
                            moments_vector[3,3],moments_vector[4,3],moments_vector[17,3]]).reshape((6,1))

error_port = np.array([moments_vector[5,3],moments_vector[6,3],moments_vector[7,3],
                            moments_vector[8,3],moments_vector[9,3],moments_vector[18,3]]).reshape((6,1))

error_eteacher = np.array([moments_vector[10,3],moments_vector[11,3],moments_vector[12,3],
                                moments_vector[13,3]]).reshape((4,1))

error_estudent = np.array([moments_vector[14,3],moments_vector[15,3],moments_vector[16,3]]).reshape((3,1))







#se_gamma_0 =moments_vector[0,1]
#se_gamma_1 =se_vector[15]
#se_gamma_2 =se_vector[16]


#Effort Student (this is t_student function in utility class)
#betas_opt_t = np.array([betas_nelder[10],betas_nelder[11],betas_nelder[12],
	#betas_nelder[13]]).reshape((4,1))

#se_betas_opt_t=np.array([se_vector[10],se_vector[11],se_vector[12],se_vector[13]]).reshape((4,1))


#income_male_betas = np.array([betas_nelder[8],betas_nelder[9],
	#betas_nelder[10]]).reshape((3,1))
#c_emp_spouse = betas_nelder[11]


#sigma_income_male_betas = np.array([se_vector[8],se_vector[9],
	#se_vector[10]]).reshape((3,1))
#sigma_c_emp_spouse = se_vector[11]

#Effort teachers (this is t_test function in utility class)
#portfolio
#alphas_port = np.array([betas_nelder[0],betas_nelder[1],betas_nelder[2],
               #betas_nelder[3],betas_nelder[4]]).reshape((5,1))

#se_alphas_port = np.array([se_vector[0],se_vector[1],se_vector[2],
               #se_vector[3],se_vector[4]]).reshape((5,1))


#alphas_test = np.array([betas_nelder[5],betas_nelder[6],betas_nelder[7],
               #betas_nelder[8],betas_nelder[9]]).reshape((5,1))

#se_alphas_test = np.array([se_vector[5],se_vector[6],se_vector[7],
               #se_vector[8],se_vector[9]]).reshape((5,1))

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

#utility_list_beta = [gamma_0,gamma_1,gamma_2]
#utility_list_se = [se_gamma_0,se_gamma_1,se_gamma_2]
#utility_names = [r'gamma_0',r'gamma_1',r'gamma_2']

sim_test_t = [simulation_test[0,0],simulation_test[1,0],simulation_test[2,0],
                  simulation_test[3,0],simulation_test[4,0],simulation_test[5,0]]
data_test_t = [data_test[0,0],data_test[1,0],data_test[2,0],data_test[3,0],data_test[4,0],data_test[5,0]]
error_test_t = [error_test[0,0],error_test[1,0],error_test[2,0],error_test[3,0],error_test[4,0],error_test[5,0]]
wage_names = ['E(Test)', 'Var(Test)','Corr(Test,past portfolio)','Corr(Test,Simce)',
              'Corr(Test,Experience)','alpha_6']


sim_port_t = [simulation_port[0,0],simulation_port[1,0],simulation_port[2,0],
                  simulation_port[3,0],simulation_port[4,0],simulation_port[5,0]]
data_port_t = [data_port[0,0],data_port[1,0],data_port[2,0],data_port[3,0],data_port[4,0],data_port[5,0]]
error_port_t = [error_port[0,0],error_port[1,0],error_port[2,0],error_port[3,0],error_port[4,0],error_port[5,0]]
wage_names_port = ['E(Portfolio)', 'Var(Portfolio)','Corr(Portfolio,past portfolio)','Corr(Portfolio,Simce)',
              'Corr(Portfolio,Experience)','alpha_6']

#swage_list_beta = [income_male_betas[0,0],income_male_betas[1,0],income_male_betas[2,0],
#c_emp_spouse]
#swage_list_se = [sigma_income_male_betas[0,0],sigma_income_male_betas[1,0],sigma_income_male_betas[2,0],
#sigma_c_emp_spouse]
#swage_names = ['High school dummy', 'Constant', 'SD of error term','Employment probability']


sim_eteacher_t = [simulation_eteacher[0,0],simulation_eteacher[1,0],simulation_eteacher[2,0],
                   simulation_eteacher[3,0]]
data_eteacher_t = [data_eteacher[0,0],data_eteacher[1,0],data_eteacher[2,0],data_eteacher[3,0]]
error_eteacher_t = [error_eteacher[0,0],error_eteacher[1,0],error_eteacher[2,0],error_eteacher[3,0]]
prod_names_young = [r'gamma_00', r'gamma_01', r'gamma_02', r'gamma_03']

sim_estudent_t = [simulation_estudent[0,0],simulation_estudent[1,0],simulation_estudent[2,0]]
data_estudent_t = [data_estudent[0,0],data_estudent[1,0],data_estudent[2,0]]
error_estudent_t = [error_estudent[0,0],error_estudent[1,0],error_estudent[2,0]]
prod_names_young_t = [r'beta_00', r'beta_01', r'beta_02']


#prod_fn_common = [sigma2theta,rho_theta_epsilon,qprob]
#prod_fn_common_se = [se_sigma2theta,se_rho_theta_epsilon,se_qprob]

#prod_fn_names_common = [r'SD of shock to human capital', r'Correlation of wage and initial human capital'
#, r'Probability of free child care']

#This is for the paper
with open('D:\Git\TeacherBranch/moment_est.tex','w') as f:
	f.write(r'\begin{tabular}{lcccc}'+'\n')
	f.write(r'\hline' + '\n')
	f.write(r'Moments &  & Simulated & & Data & & S.E. data \bigstrut\\' + '\n')
	f.write(r'\hline' + '\n')
	f.write(r'\emph{A. Utility function} &       &       &       &     &      & \\' + '\n')
	for j in range(len(sim_eteacher_t)):
		f.write(prod_names_young[j]+ r' &  & ' + '{:04.3f}'.format(sim_eteacher_t[j]) +
          r' &  &  '+ '{:04.3f}'.format(data_eteacher_t[j]) +
			r' &  & '+ '{:04.3f}'.format(error_eteacher_t[j])+ r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{B. Effort Student} &       &       &       & & &  \\' + '\n')
	for j in range(len(sim_estudent_t)):
		f.write(prod_names_young_t[j]+ r' &  &  '+ '{:04.3f}'.format(sim_estudent_t[j]) +
			r' &  & '+ '{:04.3f}'.format(error_estudent_t[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{C. Effort teachers Portfolio} &       &       &       &  & & \\' + '\n')
	for j in range(len(sim_port_t)):
		f.write(wage_names_port[j]+ r' &  &  '+ '{:04.3f}'.format(sim_port_t[j]) +
          r' &  &  '+ '{:04.3f}'.format(data_port_t[j]) + 
			r' &  & '+ '{:04.3f}'.format(error_port_t[j])+ r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{D. Effort teachers Test} &       &       &       &  & & \\' + '\n')
	for j in range(len(sim_test_t)):
		f.write(wage_names[j] + r' &  &  '+ '{:04.3f}'.format(sim_test_t[j]) +
          r' &  &  '+ '{:04.3f}'.format(data_test_t[j]) +
			r' &  & '+ '{:04.3f}'.format(error_test_t[j])+ r' \\' + '\n')

	#for j in range(len(prod_list_beta_old)):
		#f.write(prod_names_old[j]  + ' (old)' + r' &  &  '+ '{:04.3f}'.format(prod_list_beta_old[j]) +
			#r' &  & '+ '{:04.3f}'.format(prod_list_se_old[j])+r' \\' + '\n')
	#for j in range(len(prod_fn_common)):
		#f.write(prod_fn_names_common[j]+r' &  &  '+ '{:04.3f}'.format(prod_fn_common[j]) +
			#r' &  & '+ '{:04.3f}'.format(prod_fn_common_se[j])+r' \\' + '\n')

	f.write(r'\hline'+'\n')
	f.write(r'\end{tabular}' + '\n')
	f.close()




